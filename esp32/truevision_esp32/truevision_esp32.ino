/*
 * TrueVision ESP32 Firmware
 *
 * Captures audio via I2S microphone (INMP441) and streams it over
 * UART to a Raspberry Pi 5 using a custom binary framing protocol.
 * A two-position physical switch toggles between AUDIO and FACE modes.
 *
 * Protocol: [0xAA][0x55][TYPE][LEN_LO][LEN_HI][DATA][CHECKSUM]
 * Packet Types: 0x01 (Audio), 0x02 (Mode Change), 0x03 (Marker)
 */

#include <driver/i2s.h>

// I2S Microphone pins
#define I2S_WS 6
#define I2S_SD 7
#define I2S_SCK 8
#define I2S_PORT I2S_NUM_0

// UART pins
#define UART_TX 17
#define UART_RX 18
#define BAUD_RATE 921600

// Switch pins
#define SWITCH_AUDIO_PIN 35
#define SWITCH_FACE_PIN 36

// LED pins
#define LED_HEARTBEAT 9
#define LED_PACKET 10

// Audio settings
#define SAMPLE_RATE 16000
#define BUFFER_SAMPLES 256

// Packet Types
#define PKT_AUDIO 0x01
#define PKT_MODE_CHANGE 0x02
#define PKT_MARKER 0x03

int32_t i2s_raw_buffer[BUFFER_SAMPLES];
int16_t audio_buffer[BUFFER_SAMPLES];

uint8_t current_mode = 0xFF; // Unknown initially
unsigned long last_heartbeat = 0;
unsigned long last_packet_led = 0;
bool heartbeat_state = false;

void setup() {
  // Debug serial
  Serial.begin(115200);
  
  // Data UART to Pi
  Serial2.begin(BAUD_RATE, SERIAL_8N1, UART_RX, UART_TX);
  
  // Setup pins
  pinMode(SWITCH_AUDIO_PIN, INPUT);
  pinMode(SWITCH_FACE_PIN, INPUT);
  pinMode(LED_HEARTBEAT, OUTPUT);
  pinMode(LED_PACKET, OUTPUT);
  
  // Initialize I2S
  i2s_config_t i2s_config = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate = SAMPLE_RATE,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
    .communication_format = I2S_COMM_FORMAT_I2S,
    .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count = 8,
    .dma_buf_len = BUFFER_SAMPLES,
    .use_apll = false,
    .tx_desc_auto_clear = false,
    .fixed_mclk = 0
  };
  
  i2s_pin_config_t pin_config = {
    .bck_io_num = I2S_SCK,
    .ws_io_num = I2S_WS,
    .data_out_num = I2S_PIN_NO_CHANGE,
    .data_in_num = I2S_SD
  };
  
  i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL);
  i2s_set_pin(I2S_PORT, &pin_config);
  
  Serial.println("TrueVision ESP32 Booted");
  
  // Send initial mode
  check_switch();
}

void loop() {
  // 1. Heartbeat LED
  if (millis() - last_heartbeat > 500) {
    heartbeat_state = !heartbeat_state;
    digitalWrite(LED_HEARTBEAT, heartbeat_state);
    last_heartbeat = millis();
  }
  
  // 2. Turn off packet LED if needed
  if (millis() - last_packet_led > 10) {
    digitalWrite(LED_PACKET, LOW);
  }
  
  // 3. Check switch state
  check_switch();
  
  // 4. Read I2S Audio
  size_t bytes_read = 0;
  esp_err_t result = i2s_read(I2S_PORT, &i2s_raw_buffer, sizeof(i2s_raw_buffer), &bytes_read, portMAX_DELAY);
  
  if (result == ESP_OK && bytes_read > 0) {
    int samples_read = bytes_read / 4; // 32-bit samples
    
    // Convert 32-bit to 16-bit
    for (int i = 0; i < samples_read; i++) {
      audio_buffer[i] = i2s_raw_buffer[i] >> 16;
    }
    
    // Send over UART
    send_packet(PKT_AUDIO, (uint8_t*)audio_buffer, samples_read * 2);
    
    // Flash packet LED
    digitalWrite(LED_PACKET, HIGH);
    last_packet_led = millis();
  }
}

void check_switch() {
  // Simple debounce (could be improved)
  int audio_val = digitalRead(SWITCH_AUDIO_PIN);
  int face_val = digitalRead(SWITCH_FACE_PIN);
  
  uint8_t new_mode = 0xFF;
  if (audio_val == HIGH) {
    new_mode = 0x00; // AUDIO
  } else if (face_val == HIGH) {
    new_mode = 0x01; // FACE
  }
  
  if (new_mode != 0xFF && new_mode != current_mode) {
    current_mode = new_mode;
    send_packet(PKT_MODE_CHANGE, &current_mode, 1);
    Serial.printf("Mode changed: %d\n", current_mode);
  }
}

void send_packet(uint8_t type, uint8_t* data, uint16_t len) {
  uint8_t header[5];
  header[0] = 0xAA;
  header[1] = 0x55;
  header[2] = type;
  header[3] = len & 0xFF;
  header[4] = (len >> 8) & 0xFF;
  
  uint8_t checksum = 0;
  for (uint16_t i = 0; i < len; i++) {
    checksum += data[i];
  }
  
  Serial2.write(header, 5);
  Serial2.write(data, len);
  Serial2.write(&checksum, 1);
}
