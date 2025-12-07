/* Edge Impulse ESP32-CAM Inference with Live Stream and Bounding Boxes
 * Author: Modified for bounding box live stream
 */

#include <Traffic_Shoot_inferencing.h>
#include "edge-impulse-sdk/dsp/image/image.hpp"
#include "esp_camera.h"
#include <WiFi.h>
#include "esp_http_server.h"

// ---------- WiFi Credentials ----------
const char* ssid = "H&M";
const char* password = "123456798";

// ---------- Camera Model ----------
#define CAMERA_MODEL_AI_THINKER

#if defined(CAMERA_MODEL_AI_THINKER)
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22
#else
#error "Camera model not selected"
#endif

// ---------- Constants ----------
#define EI_CAMERA_RAW_FRAME_BUFFER_COLS 320
#define EI_CAMERA_RAW_FRAME_BUFFER_ROWS 240
#define EI_CAMERA_FRAME_BYTE_SIZE       3

// ---------- Globals ----------
static bool debug_nn = false;
static bool is_initialised = false;
uint8_t *snapshot_buf; 
ei_impulse_result_t last_result = {0}; // store last inference result for bounding boxes

// ---------- Camera Config ----------
static camera_config_t camera_config = {
    .pin_pwdn = PWDN_GPIO_NUM,
    .pin_reset = RESET_GPIO_NUM,
    .pin_xclk = XCLK_GPIO_NUM,
    .pin_sscb_sda = SIOD_GPIO_NUM,
    .pin_sscb_scl = SIOC_GPIO_NUM,
    .pin_d7 = Y9_GPIO_NUM,
    .pin_d6 = Y8_GPIO_NUM,
    .pin_d5 = Y7_GPIO_NUM,
    .pin_d4 = Y6_GPIO_NUM,
    .pin_d3 = Y5_GPIO_NUM,
    .pin_d2 = Y4_GPIO_NUM,
    .pin_d1 = Y3_GPIO_NUM,
    .pin_d0 = Y2_GPIO_NUM,
    .pin_vsync = VSYNC_GPIO_NUM,
    .pin_href = HREF_GPIO_NUM,
    .pin_pclk = PCLK_GPIO_NUM,
    .xclk_freq_hz = 20000000,
    .ledc_timer = LEDC_TIMER_0,
    .ledc_channel = LEDC_CHANNEL_0,
    .pixel_format = PIXFORMAT_JPEG,
    .frame_size = FRAMESIZE_QVGA,
    .jpeg_quality = 12,
    .fb_count = 1,
    .fb_location = CAMERA_FB_IN_PSRAM,
    .grab_mode = CAMERA_GRAB_WHEN_EMPTY
};

// ---------- Function Prototypes ----------
bool ei_camera_init(void);
void ei_camera_deinit(void);
bool ei_camera_capture(uint32_t img_width, uint32_t img_height, uint8_t *out_buf);
static int ei_camera_get_data(size_t offset, size_t length, float *out_ptr);
void startCameraServer();

// ---------- HTTP Stream ----------
httpd_handle_t stream_httpd = NULL;

static esp_err_t stream_handler(httpd_req_t *req){
    camera_fb_t * fb = NULL;
    esp_err_t res = ESP_OK;
    char part_buf[128];

    res = httpd_resp_set_type(req, "multipart/x-mixed-replace;boundary=frame");
    if(res != ESP_OK) return res;

    while(true){
        fb = esp_camera_fb_get();
        if(!fb) continue;

        // Draw bounding boxes on frame (RGB565)
#if EI_CLASSIFIER_OBJECT_DETECTION == 1
        for(uint32_t i=0; i<last_result.bounding_boxes_count; i++){
            ei_impulse_result_bounding_box_t bb = last_result.bounding_boxes[i];
            if(bb.value == 0) continue;

            int x0 = bb.x;
            int y0 = bb.y;
            int x1 = x0 + bb.width;
            int y1 = y0 + bb.height;

            // Ensure limits
            x0 = max(0, x0); y0 = max(0, y0);
            x1 = min((int)fb->width-1, x1); 
            y1 = min((int)fb->height-1, y1);

            uint16_t *img = (uint16_t*)fb->buf;
            // Draw top/bottom lines
            for(int x=x0; x<=x1; x++){
                img[y0*fb->width + x] = 0xF800; // red
                img[y1*fb->width + x] = 0xF800;
            }
            // Draw left/right lines
            for(int y=y0; y<=y1; y++){
                img[y*fb->width + x0] = 0xF800;
                img[y*fb->width + x1] = 0xF800;
            }
        }
#endif

        // Send JPEG frame
        httpd_resp_send_chunk(req, "--frame\r\n", strlen("--frame\r\n"));
        sprintf(part_buf, "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n", fb->len);
        httpd_resp_send_chunk(req, part_buf, strlen(part_buf));
        httpd_resp_send_chunk(req, (const char*)fb->buf, fb->len);
        httpd_resp_send_chunk(req, "\r\n", 2);

        esp_camera_fb_return(fb);
    }
    return res;
}

void startCameraServer(){
    httpd_config_t config = HTTPD_DEFAULT_CONFIG();
    config.server_port = 81;

    httpd_uri_t stream_uri = {
        .uri = "/stream",
        .method = HTTP_GET,
        .handler = stream_handler,
        .user_ctx = NULL
    };

    httpd_start(&stream_httpd, &config);
    httpd_register_uri_handler(stream_httpd, &stream_uri);
}

// ---------- Setup ----------
void setup() {
    Serial.begin(115200);
    while(!Serial);

    Serial.println("Edge Impulse Inferencing Demo");

    // WiFi
    WiFi.begin(ssid, password);
    Serial.print("Connecting to WiFi");
    while(WiFi.status() != WL_CONNECTED){
        delay(500);
        Serial.print(".");
    }
    Serial.println();
    Serial.print("WiFi connected! Stream URL: http://");
    Serial.print(WiFi.localIP());
    Serial.println(":81/stream");

    // Camera
    if(!ei_camera_init()){
        ei_printf("Failed to initialize Camera!\r\n");
    } else {
        startCameraServer();
        Serial.println("Camera stream server started");
    }

    ei_printf("\nStarting continuous inference in 2 seconds...\n");
    ei_sleep(2000);
}

// ---------- Loop ----------
void loop() {
    if(ei_sleep(5) != EI_IMPULSE_OK) return;

    snapshot_buf = (uint8_t*)malloc(EI_CAMERA_RAW_FRAME_BUFFER_COLS * EI_CAMERA_RAW_FRAME_BUFFER_ROWS * EI_CAMERA_FRAME_BYTE_SIZE);
    if(snapshot_buf == nullptr){
        ei_printf("ERR: Failed to allocate snapshot buffer!\n");
        return;
    }

    ei::signal_t signal;
    signal.total_length = EI_CLASSIFIER_INPUT_WIDTH * EI_CLASSIFIER_INPUT_HEIGHT;
    signal.get_data = &ei_camera_get_data;

    if(!ei_camera_capture(EI_CLASSIFIER_INPUT_WIDTH, EI_CLASSIFIER_INPUT_HEIGHT, snapshot_buf)){
        ei_printf("Failed to capture image\r\n");
        free(snapshot_buf);
        return;
    }

    // Run classifier
    EI_IMPULSE_ERROR err = run_classifier(&signal, &last_result, debug_nn);
    if(err != EI_IMPULSE_OK){
        ei_printf("ERR: Failed to run classifier (%d)\n", err);
        free(snapshot_buf);
        return;
    }

    ei_printf("Predictions (DSP: %d ms., Classification: %d ms., Anomaly: %d ms.):\n",
                last_result.timing.dsp, last_result.timing.classification, last_result.timing.anomaly);

#if EI_CLASSIFIER_OBJECT_DETECTION == 1
    ei_printf("Bounding Boxes:\n");
    for(uint32_t i=0; i<last_result.bounding_boxes_count; i++){
        ei_impulse_result_bounding_box_t bb = last_result.bounding_boxes[i];
        if(bb.value == 0) continue;
        ei_printf(" %s: %f [x:%u y:%u w:%u h:%u]\n", bb.label, bb.value, bb.x, bb.y, bb.width, bb.height);
    }
#else
    ei_printf("Predictions:\n");
    for(uint16_t i=0; i<EI_CLASSIFIER_LABEL_COUNT; i++){
        ei_printf("  %s: %.5f\n", ei_classifier_inferencing_categories[i], last_result.classification[i].value);
    }
#endif

#if EI_CLASSIFIER_HAS_ANOMALY
    ei_printf("Anomaly prediction: %.3f\n", last_result.anomaly);
#endif

    free(snapshot_buf);
}

// ---------- Camera Functions ----------
bool ei_camera_init(void) {
    if(is_initialised) return true;

    esp_err_t err = esp_camera_init(&camera_config);
    if(err != ESP_OK){
        Serial.printf("Camera init failed with error 0x%x\n", err);
        return false;
    }

    sensor_t * s = esp_camera_sensor_get();
    if(s->id.PID == OV3660_PID){
        s->set_vflip(s,1);
        s->set_brightness(s,1);
        s->set_saturation(s,0);
    }

    is_initialised = true;
    return true;
}

void ei_camera_deinit(void){
    esp_camera_deinit();
    is_initialised = false;
}

bool ei_camera_capture(uint32_t img_width, uint32_t img_height, uint8_t *out_buf){
    if(!is_initialised){
        ei_printf("ERR: Camera is not initialized\r\n");
        return false;
    }

    camera_fb_t *fb = esp_camera_fb_get();
    if(!fb){
        ei_printf("Camera capture failed\n");
        return false;
    }

    bool converted = fmt2rgb888(fb->buf, fb->len, PIXFORMAT_JPEG, snapshot_buf);
    esp_camera_fb_return(fb);

    if(!converted){
        ei_printf("Conversion failed\n");
        return false;
    }

    if((img_width != EI_CAMERA_RAW_FRAME_BUFFER_COLS) || (img_height != EI_CAMERA_RAW_FRAME_BUFFER_ROWS)){
        ei::image::processing::crop_and_interpolate_rgb888(
            out_buf,
            EI_CAMERA_RAW_FRAME_BUFFER_COLS,
            EI_CAMERA_RAW_FRAME_BUFFER_ROWS,
            out_buf,
            img_width,
            img_height
        );
    }
    return true;
}

static int ei_camera_get_data(size_t offset, size_t length, float *out_ptr){
    size_t pixel_ix = offset*3;
    for(size_t i=0;i<length;i++){
        out_ptr[i] = (snapshot_buf[pixel_ix+2]<<16)+(snapshot_buf[pixel_ix+1]<<8)+snapshot_buf[pixel_ix];
        pixel_ix+=3;
    }
    return 0;
}

#if !defined(EI_CLASSIFIER_SENSOR) || EI_CLASSIFIER_SENSOR != EI_CLASSIFIER_SENSOR_CAMERA
#error "Invalid model for current sensor"
#endif
