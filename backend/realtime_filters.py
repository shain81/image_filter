import cv2
import numpy as np
from filters import ImageFilters
import base64
import time
import os
from PIL import Image
import io


class RealtimeFilterProcessor:
    def __init__(self):
        self.image_filters = ImageFilters()
        self.current_filter = 'original'
        self.filter_params = {}
        self.fps = 30
        self.last_process_time = 0
        
        # حذف frame skip برای real-time صاف
        # self.frame_skip = 2  
        # self.frame_count = 0

    def set_filter(self, filter_name, params=None):
        """تنظیم فیلتر فعلی"""
        self.current_filter = filter_name
        self.filter_params = params or {}
        print(f"Filter set to: {filter_name}")  # Debug

    def process_frame(self, frame_data):
        """پردازش یک فریم ویدیو - بهینه شده"""
        try:
            # بررسی وقت برای کنترل FPS
            current_time = time.time()
            if current_time - self.last_process_time < 1.0/30:  # 30 FPS max
                return None
            self.last_process_time = current_time

            # Decode base64 image
            if ',' in frame_data:
                image_data = base64.b64decode(frame_data.split(',')[1])
            else:
                image_data = base64.b64decode(frame_data)
                
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                print("Failed to decode image")
                return None

            # تبدیل BGR به RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # اعمال فیلتر مستقیم بدون فایل موقت
            if self.current_filter == 'original':
                filtered = img_rgb
            else:
                filtered = self.apply_filter_direct(img_rgb, self.current_filter, self.filter_params)

            # تبدیل به BGR برای encoding
            filtered_bgr = cv2.cvtColor(filtered, cv2.COLOR_RGB2BGR)

            # Encode به base64 با کیفیت مناسب
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
            _, buffer = cv2.imencode('.jpg', filtered_bgr, encode_param)
            frame_base64 = base64.b64encode(buffer).decode('utf-8')

            return f"data:image/jpeg;base64,{frame_base64}"

        except Exception as e:
            print(f"Error processing frame: {str(e)}")
            return None

    def apply_filter_direct(self, img_rgb, filter_name, params):
        """اعمال فیلتر مستقیم روی numpy array - بدون فایل موقت"""
        try:
            # فیلترهای ساده که نیاز به فایل ندارند
            if filter_name == 'grayscale':
                gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
                return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            
            elif filter_name == 'sepia':
                sepia_matrix = np.array([[0.393, 0.769, 0.189],
                                         [0.349, 0.686, 0.168],
                                         [0.272, 0.534, 0.131]])
                sepia = cv2.transform(img_rgb, sepia_matrix)
                return np.clip(sepia, 0, 255).astype(np.uint8)
            
            elif filter_name == 'negative':
                return 255 - img_rgb
            
            elif filter_name == 'warm':
                warming = np.array([[1.2, 0, 0],
                                    [0, 1.0, 0],
                                    [0, 0, 0.8]])
                warmed = cv2.transform(img_rgb, warming)
                return np.clip(warmed, 0, 255).astype(np.uint8)
            
            elif filter_name == 'cool':
                cooling = np.array([[0.8, 0, 0],
                                    [0, 1.0, 0],
                                    [0, 0, 1.2]])
                cooled = cv2.transform(img_rgb, cooling)
                return np.clip(cooled, 0, 255).astype(np.uint8)
            
            elif filter_name == 'brightness':
                factor = params.get('factor', 1.5)
                bright = img_rgb.astype(np.float32) * factor
                return np.clip(bright, 0, 255).astype(np.uint8)
            
            elif filter_name == 'contrast':
                factor = params.get('factor', 1.5)
                contrast = cv2.convertScaleAbs(img_rgb, alpha=factor, beta=0)
                return contrast
            
            elif filter_name == 'gaussian_blur':
                kernel_size = params.get('kernel_size', 15)
                # اطمینان از اینکه kernel_size فرد باشد
                if kernel_size % 2 == 0:
                    kernel_size += 1
                return cv2.GaussianBlur(img_rgb, (kernel_size, kernel_size), 0)
            
            elif filter_name == 'sharpen':
                kernel = np.array([[0, -1, 0],
                                   [-1, 5, -1],
                                   [0, -1, 0]])
                return cv2.filter2D(img_rgb, -1, kernel)
            
            elif filter_name == 'emboss':
                kernel = np.array([[-2, -1, 0],
                                   [-1, 1, 1],
                                   [0, 1, 2]])
                return cv2.filter2D(img_rgb, -1, kernel)
            
            elif filter_name == 'vintage':
                # نسخه ساده‌شده vintage
                vintage_matrix = np.array([[0.5, 0.4, 0.1],
                                           [0.3, 0.5, 0.2],
                                           [0.2, 0.3, 0.5]])
                vintage = cv2.transform(img_rgb, vintage_matrix)
                return np.clip(vintage, 0, 255).astype(np.uint8)
            
            elif filter_name == 'cyberpunk':
                cyber = img_rgb.copy()
                cyber[:, :, 0] = np.clip(cyber[:, :, 0] * 1.5, 0, 255)  # Red
                cyber[:, :, 2] = np.clip(cyber[:, :, 2] * 1.8, 0, 255)  # Blue
                cyber = cv2.addWeighted(cyber, 1.5, np.zeros(cyber.shape, cyber.dtype), 0, -50)
                return np.clip(cyber, 0, 255).astype(np.uint8)
            
            elif filter_name == 'sunset':
                sunset_matrix = np.array([[1.2, 0.1, 0],
                                          [0, 0.8, 0],
                                          [0, 0.2, 0.9]])
                sunset = cv2.transform(img_rgb, sunset_matrix)
                return np.clip(sunset, 0, 255).astype(np.uint8)
            
            elif filter_name == 'night':
                night = img_rgb.astype(np.float32)
                night[:, :, 0] = night[:, :, 0] * 0.5  # Red
                night[:, :, 1] = night[:, :, 1] * 0.6  # Green
                night[:, :, 2] = night[:, :, 2] * 0.9  # Blue
                return np.clip(night, 0, 255).astype(np.uint8)
            
            elif filter_name == 'autumn':
                autumn_matrix = np.array([[1.2, 0.3, 0],
                                          [0, 0.8, 0],
                                          [0, 0, 0.5]])
                autumn = cv2.transform(img_rgb, autumn_matrix)
                return np.clip(autumn, 0, 255).astype(np.uint8)
            
            elif filter_name == 'spring':
                spring_matrix = np.array([[1.0, 0.2, 0],
                                          [0, 1.2, 0.1],
                                          [0, 0.1, 1.0]])
                spring = cv2.transform(img_rgb, spring_matrix)
                return np.clip(spring, 0, 255).astype(np.uint8)
            
            elif filter_name == 'purple_haze':
                purple = img_rgb.copy()
                purple[:, :, 0] = np.clip(purple[:, :, 0] * 1.2, 0, 255)  # Red
                purple[:, :, 2] = np.clip(purple[:, :, 2] * 1.5, 0, 255)  # Blue
                fog = np.ones_like(purple) * [150, 100, 200]
                result = cv2.addWeighted(purple, 0.7, fog, 0.3, 0)
                return result.astype(np.uint8)
            
            elif filter_name == 'golden_hour':
                golden = img_rgb.astype(np.float32)
                golden[:, :, 0] = np.clip(golden[:, :, 0] * 1.3, 0, 255)  # Red
                golden[:, :, 1] = np.clip(golden[:, :, 1] * 1.1, 0, 255)  # Green
                golden[:, :, 2] = golden[:, :, 2] * 0.7  # Blue
                golden = cv2.addWeighted(golden, 1.0, np.ones(golden.shape) * [30, 20, 0], 0.2, 0)
                return np.clip(golden, 0, 255).astype(np.uint8)
            
            elif filter_name == 'neon':
                hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
                hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 2.0, 0, 255)  # Saturation
                hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.2, 0, 255)  # Value
                neon = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
                blur = cv2.GaussianBlur(neon, (21, 21), 0)
                neon = cv2.addWeighted(neon, 0.7, blur, 0.3, 0)
                return neon
            
            elif filter_name == 'pastel':
                hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
                hsv[:, :, 1] = hsv[:, :, 1] * 0.5  # کاهش اشباع
                hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.3, 0, 255)  # افزایش روشنایی
                pastel = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
                color_overlay = np.ones_like(pastel) * [240, 220, 230]
                result = cv2.addWeighted(pastel, 0.8, color_overlay, 0.2, 0)
                return result.astype(np.uint8)
            
            elif filter_name == 'hue_shift':
                shift = params.get('shift', 30)
                hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
                hsv[:, :, 0] = (hsv[:, :, 0] + shift) % 180
                return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
            
            elif filter_name == 'saturation':
                factor = params.get('factor', 1.5)
                hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
                hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
                return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
            
            elif filter_name == 'bw_high_contrast':
                gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
                enhanced = cv2.convertScaleAbs(gray, alpha=1.5, beta=-50)
                return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
            
            elif filter_name == 'bw_low_contrast':
                gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
                low_contrast = cv2.convertScaleAbs(gray, alpha=0.5, beta=50)
                return cv2.cvtColor(low_contrast, cv2.COLOR_GRAY2RGB)
            
            elif filter_name == 'bw_red_filter':
                weights = np.array([0.5, 0.3, 0.2])
                gray = np.dot(img_rgb, weights)
                return cv2.cvtColor(gray.astype(np.uint8), cv2.COLOR_GRAY2RGB)
            
            elif filter_name == 'bw_green_filter':
                weights = np.array([0.2, 0.6, 0.2])
                gray = np.dot(img_rgb, weights)
                return cv2.cvtColor(gray.astype(np.uint8), cv2.COLOR_GRAY2RGB)
            
            elif filter_name == 'bw_blue_filter':
                weights = np.array([0.2, 0.3, 0.5])
                gray = np.dot(img_rgb, weights)
                return cv2.cvtColor(gray.astype(np.uint8), cv2.COLOR_GRAY2RGB)
            
            elif filter_name == 'film_noir':
                gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
                noir = cv2.convertScaleAbs(gray, alpha=2.0, beta=-100)
                # vignette ساده
                rows, cols = noir.shape
                kernel_x = cv2.getGaussianKernel(cols, cols // 3)
                kernel_y = cv2.getGaussianKernel(rows, rows // 3)
                kernel = kernel_y * kernel_x.T
                mask = kernel / kernel.max()
                noir = noir * mask
                return cv2.cvtColor(noir.astype(np.uint8), cv2.COLOR_GRAY2RGB)
            
            elif filter_name == 'thermal':
                gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
                thermal = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
                return cv2.cvtColor(thermal, cv2.COLOR_BGR2RGB)
            
            elif filter_name == 'xray':
                xray = 255 - img_rgb
                gray = cv2.cvtColor(xray, cv2.COLOR_RGB2GRAY)
                gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=0)
                xray = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
                xray[:, :, 1] = np.clip(xray[:, :, 1] * 1.2, 0, 255)  # Green
                xray[:, :, 2] = np.clip(xray[:, :, 2] * 1.4, 0, 255)  # Blue
                return xray
            
            elif filter_name == 'matrix':
                matrix = img_rgb.copy()
                matrix[:, :, 0] = 0  # حذف قرمز
                matrix[:, :, 2] = matrix[:, :, 2] * 0.3  # کاهش آبی
                matrix[:, :, 1] = np.clip(matrix[:, :, 1] * 1.5, 0, 255)  # تقویت سبز
                return np.clip(matrix, 0, 255).astype(np.uint8)
            
            elif filter_name == 'vignette':
                strength = params.get('strength', 0.8)
                rows, cols = img_rgb.shape[:2]
                kernel_x = cv2.getGaussianKernel(cols, cols // 2)
                kernel_y = cv2.getGaussianKernel(rows, rows // 2)
                kernel = kernel_y * kernel_x.T
                mask = kernel / kernel.max()
                mask = 1 - (1 - mask) * strength
                result = img_rgb.copy()
                for i in range(3):
                    result[:, :, i] = result[:, :, i] * mask
                return result.astype(np.uint8)
            
            elif filter_name == 'pixelate':
                pixel_size = params.get('pixel_size', 10)
                height, width = img_rgb.shape[:2]
                small = cv2.resize(img_rgb, (width // pixel_size, height // pixel_size),
                                   interpolation=cv2.INTER_NEAREST)
                pixelated = cv2.resize(small, (width, height),
                                       interpolation=cv2.INTER_NEAREST)
                return pixelated
            
            elif filter_name == 'mosaic':
                block_size = params.get('block_size', 10)
                rows, cols = img_rgb.shape[:2]
                result = np.zeros_like(img_rgb)
                for y in range(0, rows, block_size):
                    for x in range(0, cols, block_size):
                        block = img_rgb[y:min(y + block_size, rows), x:min(x + block_size, cols)]
                        if block.size > 0:
                            avg_color = np.mean(block, axis=(0, 1))
                            result[y:min(y + block_size, rows), x:min(x + block_size, cols)] = avg_color
                return result.astype(np.uint8)
            
            else:
                # برای فیلترهای پیچیده‌تر، از روش قدیمی استفاده کن
                return self.apply_filter_with_file(img_rgb, filter_name, params)

        except Exception as e:
            print(f"Error in apply_filter_direct: {e}")
            return img_rgb

    def apply_filter_with_file(self, img_rgb, filter_name, params):
        """برای فیلترهای پیچیده که نیاز به فایل دارند"""
        try:
            # تبدیل به PIL Image
            pil_image = Image.fromarray(img_rgb)

            # ذخیره موقت در حافظه
            img_buffer = io.BytesIO()
            pil_image.save(img_buffer, format='PNG')
            img_buffer.seek(0)

            # ایجاد مسیر فایل موقت
            temp_filename = f'temp_realtime_{time.time()}.png'
            with open(temp_filename, 'wb') as f:
                f.write(img_buffer.getvalue())

            try:
                # اعمال فیلتر
                filtered = self.image_filters.apply_filter(temp_filename, filter_name, **params)
                return filtered
            finally:
                # حذف فایل موقت
                if os.path.exists(temp_filename):
                    os.remove(temp_filename)

        except Exception as e:
            print(f"Error in apply_filter_with_file: {e}")
            return img_rgb


# فیلترهای بهینه شده برای real-time
REALTIME_OPTIMIZED_FILTERS = [
    'original', 'grayscale', 'sepia', 'negative', 'brightness', 'contrast',
    'warm', 'cool', 'vintage', 'cyberpunk', 'sunset', 'night', 'autumn', 'spring',
    'purple_haze', 'golden_hour', 'neon', 'pastel', 'hue_shift', 'saturation',
    'bw_high_contrast', 'bw_low_contrast', 'bw_red_filter', 'bw_green_filter',
    'bw_blue_filter', 'film_noir', 'thermal', 'xray', 'matrix',
    'vignette', 'pixelate', 'mosaic', 'gaussian_blur', 'sharpen', 'emboss'
]

# فیلترهای سنگین که نیاز به فایل دارند
HEAVY_FILTERS = [
    'oil_painting', 'watercolor', 'impressionist', 'pointillism',
    'stained_glass', 'radial_blur', 'zoom_blur', 'crystallize',
    'hdr', 'clahe', 'denoise', 'surface_blur', 'cartoon', 'pop_art',
    'comic_book', 'pencil_sketch', 'colored_pencil'
]

