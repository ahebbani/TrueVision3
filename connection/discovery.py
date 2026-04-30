"""
TrueVision — Server Discovery

Discovers and monitors the TrueVision server via explicit URL or
mDNS/Zeroconf autodiscovery.
"""

import logging
import socket
import threading
import time
from typing import Optional

import requests
from zeroconf import ServiceBrowser, Zeroconf

logger = logging.getLogger(__name__)


class ServerDiscoveryListener:
    def __init__(self):
        self.server_url = None
        self.event = threading.Event()

    def remove_service(self, zeroconf, type, name):
        if self.server_url:
            logger.info(f"mDNS: Server {name} disappeared.")
            self.server_url = None

    def add_service(self, zeroconf, type, name):
        info = zeroconf.get_service_info(type, name)
        if info:
            addresses = [socket.inet_ntoa(addr) for addr in info.addresses]
            if addresses:
                ip = addresses[0]
                port = info.port
                self.server_url = f"http://{ip}:{port}"
                logger.info(f"mDNS: Discovered TrueVision server at {self.server_url}")
                self.event.set()

    def update_service(self, zeroconf, type, name):
        pass


class ServerConnection:
    def __init__(self, explicit_url: str = ""):
        self.explicit_url = explicit_url.strip()
        self.current_url: Optional[str] = None
        self._is_available = False
        self.lock = threading.Lock()
        
        self.running = False
        self.thread: Optional[threading.Thread] = None
        
        self.zeroconf = None
        self.browser = None
        self.mdns_listener = None

    def start(self):
        if self.running:
            return
            
        self.running = True
        
        # Initial resolution
        if self.explicit_url:
            logger.info(f"Using explicit server URL: {self.explicit_url}")
            self.current_url = self.explicit_url
            if not self.current_url.startswith("http"):
                self.current_url = "http://" + self.current_url
        else:
            logger.info("No explicit server URL provided. Starting mDNS discovery...")
            self.zeroconf = Zeroconf()
            self.mdns_listener = ServerDiscoveryListener()
            self.browser = ServiceBrowser(self.zeroconf, "_truevision._tcp.local.", self.mdns_listener)
            
        self.thread = threading.Thread(target=self._health_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.browser:
            self.browser.cancel()
        if self.zeroconf:
            self.zeroconf.close()
        if self.thread:
            self.thread.join(timeout=2.0)

    @property
    def is_available(self) -> bool:
        with self.lock:
            return self._is_available

    @property
    def url(self) -> Optional[str]:
        with self.lock:
            return self.current_url

    def _health_loop(self):
        # Startup retries
        retries = 3
        while retries > 0 and self.running:
            self._check_health()
            if self.is_available:
                break
            retries -= 1
            time.sleep(2.0)
            
        if not self.is_available:
            logger.warning("Server not available at startup. Will continue polling in background.")

        # Background polling every 5s
        while self.running:
            time.sleep(5.0)
            if not self.running:
                break
                
            # If mDNS, update current_url
            if not self.explicit_url and self.mdns_listener:
                new_url = self.mdns_listener.server_url
                with self.lock:
                    self.current_url = new_url
                    
            self._check_health()

    def _check_health(self):
        with self.lock:
            url = self.current_url
            
        if not url:
            self._set_available(False)
            return
            
        try:
            resp = requests.get(f"{url}/health", timeout=2.0)
            if resp.status_code == 200:
                self._set_available(True)
            else:
                self._set_available(False)
        except requests.RequestException:
            self._set_available(False)

    def _set_available(self, status: bool):
        with self.lock:
            if self._is_available != status:
                self._is_available = status
                state_str = "CONNECTED" if status else "DISCONNECTED"
                logger.info(f"Server state changed to: {state_str}")
