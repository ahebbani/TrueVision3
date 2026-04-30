"""
TrueVision — ESP32 Serial Audio Receiver

Daemon thread that reads the UART serial port, parses the custom binary
framing protocol, and stores audio in a thread-safe ring buffer.

Protocol: [0xAA][0x55][TYPE(1)][LEN_LO][LEN_HI][DATA(LEN)][CHECKSUM]
"""

import logging
import threading
import time
from typing import Callable, Optional

import numpy as np
import serial

logger = logging.getLogger(__name__)

# Packet Types
PKT_AUDIO = 0x01
PKT_MODE_CHANGE = 0x02
PKT_MARKER = 0x03


class SerialReceiver:
    _instances = {}
    _instances_lock = threading.Lock()

    @classmethod
    def get_instance(cls, port: str, baud_rate: int = 921600) -> "SerialReceiver":
        with cls._instances_lock:
            if port not in cls._instances:
                cls._instances[port] = cls(port, baud_rate)
            return cls._instances[port]

    def __init__(self, port: str, baud_rate: int):
        self.port = port
        self.baud_rate = baud_rate
        self.serial: Optional[serial.Serial] = None
        self.running = False
        self.thread: Optional[threading.Thread] = None

        # Callbacks
        self.on_mode_change: Optional[Callable[[int], None]] = None
        self.on_marker: Optional[Callable[[], None]] = None

        # Ring buffer for audio (16kHz, 16-bit mono)
        self.sample_rate = 16000
        self.buffer_capacity_sec = 60
        self.capacity_samples = self.sample_rate * self.buffer_capacity_sec
        self.audio_buffer = np.zeros(self.capacity_samples, dtype=np.int16)
        self.write_idx = 0
        self.total_samples_received = 0
        self.buffer_lock = threading.Lock()

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()
        logger.info(f"Started SerialReceiver on {self.port}")

    def stop(self):
        self.running = False
        if self.serial:
            try:
                self.serial.cancel_read()
            except Exception:
                pass
        if self.thread:
            self.thread.join(timeout=2.0)
        logger.info("Stopped SerialReceiver")

    def _connect(self) -> bool:
        try:
            self.serial = serial.Serial(
                port=self.port,
                baudrate=self.baud_rate,
                timeout=1.0
            )
            return True
        except serial.SerialException as e:
            logger.error(f"Failed to open serial port {self.port}: {e}")
            return False

    def _read_loop(self):
        backoff = 1.0
        while self.running:
            if self.serial is None or not self.serial.is_open:
                if not self._connect():
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 10.0)
                    continue
                backoff = 1.0
                logger.info(f"Connected to ESP32 on {self.port}")

            try:
                self._parse_protocol()
            except serial.SerialException as e:
                logger.error(f"Serial read error: {e}")
                self.serial.close()
            except Exception as e:
                logger.error(f"Unexpected error in serial read loop: {e}")

    def _parse_protocol(self):
        # State machine parser for [0xAA][0x55][TYPE][LEN_LO][LEN_HI][DATA][CHECKSUM]
        state = 0
        pkt_type = 0
        pkt_len = 0
        
        while self.running and self.serial and self.serial.is_open:
            if state == 0:
                b = self.serial.read(1)
                if not b: continue
                if b[0] == 0xAA: state = 1
                
            elif state == 1:
                b = self.serial.read(1)
                if not b: state = 0; continue
                if b[0] == 0x55: state = 2
                else: state = 0
                
            elif state == 2:
                b = self.serial.read(1)
                if not b: state = 0; continue
                pkt_type = b[0]
                state = 3
                
            elif state == 3:
                b = self.serial.read(1)
                if not b: state = 0; continue
                pkt_len = b[0]
                state = 4
                
            elif state == 4:
                b = self.serial.read(1)
                if not b: state = 0; continue
                pkt_len |= (b[0] << 8)
                state = 5
                
            elif state == 5:
                data = self.serial.read(pkt_len)
                if len(data) != pkt_len:
                    state = 0; continue
                state = 6
                
            elif state == 6:
                b = self.serial.read(1)
                if not b: state = 0; continue
                checksum = b[0]
                
                # Verify checksum
                calc_checksum = sum(data) & 0xFF
                if checksum == calc_checksum:
                    self._handle_packet(pkt_type, data)
                else:
                    logger.warning(f"Checksum mismatch. Expected {calc_checksum}, got {checksum}")
                
                state = 0

    def _handle_packet(self, pkt_type: int, data: bytes):
        if pkt_type == PKT_AUDIO:
            # Data is int16 little-endian
            samples = np.frombuffer(data, dtype=np.int16)
            self._write_audio(samples)
        elif pkt_type == PKT_MODE_CHANGE:
            if len(data) == 1 and self.on_mode_change:
                self.on_mode_change(data[0])
        elif pkt_type == PKT_MARKER:
            if self.on_marker:
                self.on_marker()

    def _write_audio(self, samples: np.ndarray):
        n = len(samples)
        with self.buffer_lock:
            if n >= self.capacity_samples:
                self.audio_buffer[:] = samples[-self.capacity_samples:]
                self.write_idx = 0
            else:
                end_idx = self.write_idx + n
                if end_idx <= self.capacity_samples:
                    self.audio_buffer[self.write_idx:end_idx] = samples
                else:
                    part1 = self.capacity_samples - self.write_idx
                    part2 = n - part1
                    self.audio_buffer[self.write_idx:] = samples[:part1]
                    self.audio_buffer[:part2] = samples[part1:]
                self.write_idx = end_idx % self.capacity_samples
            self.total_samples_received += n

    def clear_buffer(self):
        with self.buffer_lock:
            self.total_samples_received = 0
            self.write_idx = 0

    def get_last_n_seconds(self, seconds: float) -> np.ndarray:
        samples_needed = int(seconds * self.sample_rate)
        with self.buffer_lock:
            samples_available = min(self.total_samples_received, self.capacity_samples)
            if samples_available == 0:
                return np.array([], dtype=np.int16)
                
            samples_to_read = min(samples_needed, samples_available)
            
            if self.write_idx >= samples_to_read:
                return self.audio_buffer[self.write_idx - samples_to_read : self.write_idx].copy()
            else:
                part1_size = samples_to_read - self.write_idx
                part1 = self.audio_buffer[-part1_size:]
                part2 = self.audio_buffer[:self.write_idx]
                return np.concatenate((part1, part2))

    def get_all_audio(self) -> np.ndarray:
        with self.buffer_lock:
            samples_available = min(self.total_samples_received, self.capacity_samples)
            return self.get_last_n_seconds(samples_available / self.sample_rate)
