import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import io
import math


class ImageFilters:
    def __init__(self):
        self.filters = {
            # فیلترهای اصلی
            'original': self.original,
            'grayscale': self.grayscale,
            'sepia': self.sepia,
            'negative': self.negative,
            'brightness': self.brightness,
            'contrast': self.contrast,

            # فیلترهای رنگی جدید
            'warm': self.warm_filter,
            'cool': self.cool_filter,
            'vintage': self.vintage,
            'cyberpunk': self.cyberpunk,
            'sunset': self.sunset,
            'night': self.night,
            'autumn': self.autumn,
            'spring': self.spring,
            'purple_haze': self.purple_haze,
            'golden_hour': self.golden_hour,
            'neon': self.neon,
            'pastel': self.pastel,
            'duotone': self.duotone,
            'tritone': self.tritone,
            'hue_shift': self.hue_shift,
            'saturation': self.saturation,
            'vibrance': self.vibrance,
            'color_balance': self.color_balance,

            # فیلترهای سیاه و سفید پیشرفته
            'bw_high_contrast': self.bw_high_contrast,
            'bw_low_contrast': self.bw_low_contrast,
            'bw_red_filter': self.bw_red_filter,
            'bw_green_filter': self.bw_green_filter,
            'bw_blue_filter': self.bw_blue_filter,
            'bw_orange_filter': self.bw_orange_filter,
            'bw_yellow_filter': self.bw_yellow_filter,
            'ansel_adams': self.ansel_adams,
            'film_noir': self.film_noir,
            'infrared': self.infrared,

            # فیلترهای محو
            'gaussian_blur': self.gaussian_blur,
            'motion_blur': self.motion_blur,
            'box_blur': self.box_blur,
            'radial_blur': self.radial_blur,
            'zoom_blur': self.zoom_blur,
            'tilt_shift': self.tilt_shift,
            'lens_blur': self.lens_blur,
            'surface_blur': self.surface_blur,

            # فیلترهای تشخیص لبه
            'sobel': self.sobel,
            'canny': self.canny,
            'laplacian': self.laplacian,
            'prewitt': self.prewitt,
            'roberts': self.roberts,

            # فیلترهای هنری
            'emboss': self.emboss,
            'oil_painting': self.oil_painting,
            'pencil_sketch': self.pencil_sketch,
            'colored_pencil': self.colored_pencil,
            'cartoon': self.cartoon,
            'watercolor': self.watercolor,
            'pointillism': self.pointillism,
            'impressionist': self.impressionist,
            'pop_art': self.pop_art,
            'comic_book': self.comic_book,
            'mosaic': self.mosaic,
            'stained_glass': self.stained_glass,

            # فیلترهای قدیمی و فیلم
            'vintage_film': self.vintage_film,
            'kodachrome': self.kodachrome,
            'polaroid': self.polaroid,
            'lomo': self.lomo,
            'cross_process': self.cross_process,
            'faded_film': self.faded_film,
            'old_photo': self.old_photo,
            'daguerreotype': self.daguerreotype,

            # فیلترهای خاص
            'hdr': self.hdr,
            'glamour': self.glamour,
            'dramatic': self.dramatic,
            'dreamy': self.dreamy,
            'ethereal': self.ethereal,
            'grunge': self.grunge,
            'rainbow': self.rainbow,
            'thermal': self.thermal,
            'xray': self.xray,
            'matrix': self.matrix,

            # فیلترهای تکنیکی
            'sharpen': self.sharpen,
            'denoise': self.denoise,
            'histogram_eq': self.histogram_equalization,
            'clahe': self.clahe,
            'unsharp_mask': self.unsharp_mask,
            'edge_preserve': self.edge_preserve_smooth,

            # فیلترهای تغییر شکل و دیستورشن
            'fisheye': self.fisheye,
            'barrel_distortion': self.barrel_distortion,
            'pincushion': self.pincushion,
            'wave': self.wave_distortion,
            'swirl': self.swirl,
            'pixelate': self.pixelate,
            'crystallize': self.crystallize,

            # فیلترهای نور و سایه
            'vignette': self.vignette,
            'light_leak': self.light_leak,
            'lens_flare': self.lens_flare,
            'sun_rays': self.sun_rays,
            'soft_light': self.soft_light,
            'hard_light': self.hard_light,
        }

    def apply_filter(self, image_path, filter_name, **params):
        """اعمال فیلتر انتخاب شده روی تصویر"""
        if filter_name not in self.filters:
            raise ValueError(f"فیلتر '{filter_name}' وجود ندارد")

        # خواندن تصویر
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # اعمال فیلتر
        result = self.filters[filter_name](img_rgb, **params)

        return result

    def original(self, img, **params):
        """تصویر اصلی بدون تغییر"""
        return img

    # ==================== فیلترهای پایه ====================

    def grayscale(self, img, **params):
        """تبدیل به سیاه و سفید"""
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    def sepia(self, img, **params):
        """افکت سپیا (قدیمی)"""
        sepia_matrix = np.array([[0.393, 0.769, 0.189],
                                 [0.349, 0.686, 0.168],
                                 [0.272, 0.534, 0.131]])
        sepia = cv2.transform(img, sepia_matrix)
        sepia = np.clip(sepia, 0, 255)
        return sepia.astype(np.uint8)

    def negative(self, img, **params):
        """معکوس کردن رنگ‌ها"""
        return 255 - img

    def brightness(self, img, **params):
        """تنظیم روشنایی"""
        factor = params.get('factor', 1.5)
        pil_img = Image.fromarray(img)
        enhancer = ImageEnhance.Brightness(pil_img)
        result = enhancer.enhance(factor)
        return np.array(result)

    def contrast(self, img, **params):
        """تنظیم کنتراست"""
        factor = params.get('factor', 1.5)
        pil_img = Image.fromarray(img)
        enhancer = ImageEnhance.Contrast(pil_img)
        result = enhancer.enhance(factor)
        return np.array(result)

    # ==================== فیلترهای رنگی جدید ====================

    def warm_filter(self, img, **params):
        """فیلتر گرم"""
        warming = np.array([[1.2, 0, 0],
                            [0, 1.0, 0],
                            [0, 0, 0.8]])
        warmed = cv2.transform(img, warming)
        return np.clip(warmed, 0, 255).astype(np.uint8)

    def cool_filter(self, img, **params):
        """فیلتر سرد"""
        cooling = np.array([[0.8, 0, 0],
                            [0, 1.0, 0],
                            [0, 0, 1.2]])
        cooled = cv2.transform(img, cooling)
        return np.clip(cooled, 0, 255).astype(np.uint8)

    def vintage(self, img, **params):
        """افکت وینتیج"""
        # اضافه کردن نویز
        noise = np.random.normal(0, 5, img.shape)
        noisy = img + noise

        # اعمال فیلتر رنگی
        vintage_matrix = np.array([[0.5, 0.5, 0.1],
                                   [0.3, 0.6, 0.1],
                                   [0.2, 0.3, 0.5]])
        vintage = cv2.transform(noisy, vintage_matrix)

        # اضافه کردن vignette
        rows, cols = img.shape[:2]
        kernel_x = cv2.getGaussianKernel(cols, cols / 2)
        kernel_y = cv2.getGaussianKernel(rows, rows / 2)
        kernel = kernel_y * kernel_x.T
        mask = kernel / kernel.max()

        for i in range(3):
            vintage[:, :, i] = vintage[:, :, i] * mask

        return np.clip(vintage, 0, 255).astype(np.uint8)

    def cyberpunk(self, img, **params):
        """افکت سایبرپانک"""
        # تقویت رنگ‌های نئون
        cyber = img.copy()
        cyber[:, :, 0] = np.clip(cyber[:, :, 0] * 1.5, 0, 255)  # Red
        cyber[:, :, 2] = np.clip(cyber[:, :, 2] * 1.8, 0, 255)  # Blue

        # افزایش کنتراست
        cyber = cv2.addWeighted(cyber, 1.5, np.zeros(cyber.shape, cyber.dtype), 0, -50)

        return np.clip(cyber, 0, 255).astype(np.uint8)

    def sunset(self, img, **params):
        """افکت غروب"""
        sunset_matrix = np.array([[1.2, 0.1, 0],
                                  [0, 0.8, 0],
                                  [0, 0.2, 0.9]])
        sunset = cv2.transform(img, sunset_matrix)
        return np.clip(sunset, 0, 255).astype(np.uint8)

    def night(self, img, **params):
        """افکت شب"""
        # کاهش روشنایی و تقویت آبی
        night = img.astype(np.float32)
        night[:, :, 0] = night[:, :, 0] * 0.5  # Red
        night[:, :, 1] = night[:, :, 1] * 0.6  # Green
        night[:, :, 2] = night[:, :, 2] * 0.9  # Blue

        return np.clip(night, 0, 255).astype(np.uint8)

    def autumn(self, img, **params):
        """افکت پاییز"""
        autumn_matrix = np.array([[1.2, 0.3, 0],
                                  [0, 0.8, 0],
                                  [0, 0, 0.5]])
        autumn = cv2.transform(img, autumn_matrix)
        return np.clip(autumn, 0, 255).astype(np.uint8)

    def spring(self, img, **params):
        """افکت بهار"""
        spring_matrix = np.array([[1.0, 0.2, 0],
                                  [0, 1.2, 0.1],
                                  [0, 0.1, 1.0]])
        spring = cv2.transform(img, spring_matrix)
        return np.clip(spring, 0, 255).astype(np.uint8)

    def purple_haze(self, img, **params):
        """افکت مه بنفش"""
        purple = img.copy()
        purple[:, :, 0] = np.clip(purple[:, :, 0] * 1.2, 0, 255)  # Red
        purple[:, :, 2] = np.clip(purple[:, :, 2] * 1.5, 0, 255)  # Blue

        # اضافه کردن مه
        fog = np.ones_like(purple) * [150, 100, 200]
        result = cv2.addWeighted(purple, 0.7, fog, 0.3, 0)

        return result.astype(np.uint8)

    def golden_hour(self, img, **params):
        """افکت ساعت طلایی"""
        golden = img.astype(np.float32)
        golden[:, :, 0] = np.clip(golden[:, :, 0] * 1.3, 0, 255)  # Red
        golden[:, :, 1] = np.clip(golden[:, :, 1] * 1.1, 0, 255)  # Green
        golden[:, :, 2] = golden[:, :, 2] * 0.7  # Blue

        # اضافه کردن درخشش
        golden = cv2.addWeighted(golden, 1.0, np.ones(golden.shape) * [30, 20, 0], 0.2, 0)

        return np.clip(golden, 0, 255).astype(np.uint8)

    def neon(self, img, **params):
        """افکت نئون"""
        # افزایش اشباع رنگ
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 2.0, 0, 255)  # Saturation
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.2, 0, 255)  # Value

        neon = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        # اضافه کردن glow
        blur = cv2.GaussianBlur(neon, (21, 21), 0)
        neon = cv2.addWeighted(neon, 0.7, blur, 0.3, 0)

        return neon

    def pastel(self, img, **params):
        """افکت پاستل"""
        # کاهش اشباع و افزایش روشنایی
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] = hsv[:, :, 1] * 0.5  # کاهش اشباع
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.3, 0, 255)  # افزایش روشنایی

        pastel = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        # اضافه کردن رنگ پاستلی
        pastel_overlay = np.ones_like(pastel) * [240, 220, 230]
        result = cv2.addWeighted(pastel, 0.8, pastel_overlay, 0.2, 0)

        return result.astype(np.uint8)

    def duotone(self, img, **params):
        """افکت دو رنگ"""
        color1 = params.get('color1', [255, 100, 0])  # نارنجی
        color2 = params.get('color2', [0, 100, 255])  # آبی

        # تبدیل به grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # نرمال‌سازی
        normalized = gray / 255.0

        # ایجاد duotone
        duotone = np.zeros_like(img)
        for i in range(3):
            duotone[:, :, i] = (1 - normalized) * color1[i] + normalized * color2[i]

        return duotone.astype(np.uint8)

    def tritone(self, img, **params):
        """افکت سه رنگ"""
        # تبدیل به grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # تقسیم به سه سطح
        dark = (gray < 85).astype(np.float32)
        mid = ((gray >= 85) & (gray < 170)).astype(np.float32)
        light = (gray >= 170).astype(np.float32)

        # رنگ‌ها
        color1 = np.array([50, 0, 100])  # بنفش تیره
        color2 = np.array([200, 100, 50])  # نارنجی
        color3 = np.array([255, 200, 150])  # کرم

        # ترکیب
        result = np.zeros_like(img)
        for i in range(3):
            result[:, :, i] = dark * color1[i] + mid * color2[i] + light * color3[i]

        return result.astype(np.uint8)

    def hue_shift(self, img, **params):
        """تغییر رنگ (Hue)"""
        shift = params.get('shift', 30)

        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 0] = (hsv[:, :, 0] + shift) % 180

        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    def saturation(self, img, **params):
        """تنظیم اشباع رنگ"""
        factor = params.get('factor', 1.5)

        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)

        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    def vibrance(self, img, **params):
        """افزایش جذابیت رنگ‌ها"""
        amount = params.get('amount', 50)

        # تبدیل به LAB
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)

        # افزایش vibrance فقط برای رنگ‌های کم اشباع
        a_channel = lab[:, :, 1]
        b_channel = lab[:, :, 2]

        # محاسبه اشباع
        saturation = np.sqrt(a_channel ** 2 + b_channel ** 2)

        # اعمال vibrance بیشتر به رنگ‌های کم اشباع
        factor = 1 + (amount / 100) * (1 - saturation / 128)

        lab[:, :, 1] = np.clip(a_channel * factor, -128, 127)
        lab[:, :, 2] = np.clip(b_channel * factor, -128, 127)

        return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)

    def color_balance(self, img, **params):
        """تنظیم تعادل رنگ"""
        red_gain = params.get('red', 1.0)
        green_gain = params.get('green', 1.0)
        blue_gain = params.get('blue', 1.0)

        balanced = img.astype(np.float32)
        balanced[:, :, 0] = np.clip(balanced[:, :, 0] * red_gain, 0, 255)
        balanced[:, :, 1] = np.clip(balanced[:, :, 1] * green_gain, 0, 255)
        balanced[:, :, 2] = np.clip(balanced[:, :, 2] * blue_gain, 0, 255)

        return balanced.astype(np.uint8)

    # ==================== فیلترهای سیاه و سفید پیشرفته ====================

    def bw_high_contrast(self, img, **params):
        """سیاه و سفید با کنتراست بالا"""
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # افزایش کنتراست
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # تنظیم سطوح
        enhanced = cv2.convertScaleAbs(enhanced, alpha=1.5, beta=-50)

        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)

    def bw_low_contrast(self, img, **params):
        """سیاه و سفید با کنتراست پایین"""
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # کاهش کنتراست
        low_contrast = cv2.convertScaleAbs(gray, alpha=0.5, beta=50)

        return cv2.cvtColor(low_contrast, cv2.COLOR_GRAY2RGB)

    def bw_red_filter(self, img, **params):
        """سیاه و سفید با فیلتر قرمز"""
        # شبیه‌سازی فیلتر قرمز در عکاسی سیاه و سفید
        weights = np.array([0.5, 0.3, 0.2])
        gray = np.dot(img, weights)

        return cv2.cvtColor(gray.astype(np.uint8), cv2.COLOR_GRAY2RGB)

    def bw_green_filter(self, img, **params):
        """سیاه و سفید با فیلتر سبز"""
        weights = np.array([0.2, 0.6, 0.2])
        gray = np.dot(img, weights)

        return cv2.cvtColor(gray.astype(np.uint8), cv2.COLOR_GRAY2RGB)

    def bw_blue_filter(self, img, **params):
        """سیاه و سفید با فیلتر آبی"""
        weights = np.array([0.2, 0.3, 0.5])
        gray = np.dot(img, weights)

        return cv2.cvtColor(gray.astype(np.uint8), cv2.COLOR_GRAY2RGB)

    def bw_orange_filter(self, img, **params):
        """سیاه و سفید با فیلتر نارنجی"""
        weights = np.array([0.4, 0.4, 0.2])
        gray = np.dot(img, weights)

        return cv2.cvtColor(gray.astype(np.uint8), cv2.COLOR_GRAY2RGB)

    def bw_yellow_filter(self, img, **params):
        """سیاه و سفید با فیلتر زرد"""
        weights = np.array([0.35, 0.45, 0.2])
        gray = np.dot(img, weights)

        return cv2.cvtColor(gray.astype(np.uint8), cv2.COLOR_GRAY2RGB)

    def ansel_adams(self, img, **params):
        """سبک Ansel Adams"""
        # تبدیل به سیاه و سفید
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Zone System - تقسیم به 10 منطقه تونال
        zones = 10
        zone_values = np.linspace(0, 255, zones)

        # کوانتیزه کردن به zones
        quantized = np.zeros_like(gray)
        for i in range(zones - 1):
            mask = (gray >= zone_values[i]) & (gray < zone_values[i + 1])
            quantized[mask] = zone_values[i]

        # افزایش کنتراست در مناطق میانی
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        result = clahe.apply(quantized.astype(np.uint8))

        return cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)

    def film_noir(self, img, **params):
        """افکت فیلم نوآر"""
        # تبدیل به سیاه و سفید با کنتراست بالا
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # افزایش کنتراست شدید
        noir = cv2.convertScaleAbs(gray, alpha=2.0, beta=-100)

        # اضافه کردن grain
        noise = np.random.normal(0, 10, gray.shape)
        noir = np.clip(noir + noise, 0, 255)

        # vignette تیره
        rows, cols = noir.shape
        kernel_x = cv2.getGaussianKernel(cols, cols / 3)
        kernel_y = cv2.getGaussianKernel(rows, rows / 3)
        kernel = kernel_y * kernel_x.T
        mask = kernel / kernel.max()
        noir = noir * mask

        return cv2.cvtColor(noir.astype(np.uint8), cv2.COLOR_GRAY2RGB)

    def infrared(self, img, **params):
        """افکت مادون قرمز"""
        # تعویض کانال قرمز و سبز
        infrared = img.copy()
        infrared[:, :, 0] = img[:, :, 1]  # Red <- Green
        infrared[:, :, 1] = img[:, :, 0]  # Green <- Red

        # افزایش کنتراست
        infrared = cv2.convertScaleAbs(infrared, alpha=1.5, beta=0)

        # تبدیل به سیاه و سفید با وزن‌های خاص
        weights = np.array([0.1, 0.8, 0.1])
        gray = np.dot(infrared, weights)

        # معکوس کردن برای افکت مادون قرمز
        inverted = 255 - gray

        return cv2.cvtColor(inverted.astype(np.uint8), cv2.COLOR_GRAY2RGB)

    # ==================== فیلترهای محو پیشرفته ====================

    def gaussian_blur(self, img, **params):
        """محو گاوسی"""
        kernel_size = params.get('kernel_size', 15)
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

    def motion_blur(self, img, **params):
        """افکت محو حرکتی"""
        size = params.get('size', 15)
        angle = params.get('angle', 0)

        # ایجاد kernel حرکتی با زاویه
        kernel = np.zeros((size, size))
        center = size // 2

        # محاسبه نقاط خط با زاویه مشخص
        angle_rad = np.deg2rad(angle)
        for i in range(size):
            x = int(center + (i - center) * np.cos(angle_rad))
            y = int(center + (i - center) * np.sin(angle_rad))
            if 0 <= x < size and 0 <= y < size:
                kernel[y, x] = 1

        kernel = kernel / np.sum(kernel)
        return cv2.filter2D(img, -1, kernel)

    def box_blur(self, img, **params):
        """محو جعبه‌ای"""
        kernel_size = params.get('kernel_size', 9)
        return cv2.blur(img, (kernel_size, kernel_size))

    def radial_blur(self, img, **params):
        """محو شعاعی"""
        strength = params.get('strength', 10)

        rows, cols = img.shape[:2]
        center_x = cols // 2
        center_y = rows // 2

        result = np.zeros_like(img, dtype=np.float32)

        for angle in np.linspace(0, 2 * np.pi, strength):
            M = cv2.getRotationMatrix2D((center_x, center_y), np.degrees(angle), 1)
            rotated = cv2.warpAffine(img, M, (cols, rows))
            result += rotated

        result = result / strength
        return result.astype(np.uint8)

    def zoom_blur(self, img, **params):
        """محو زوم"""
        strength = params.get('strength', 0.2)

        rows, cols = img.shape[:2]
        center_x = cols // 2
        center_y = rows // 2

        result = img.copy().astype(np.float32)

        for i in range(1, 10):
            scale = 1 + (i * strength / 10)
            M = cv2.getRotationMatrix2D((center_x, center_y), 0, scale)

            # تنظیم مرکز
            M[0, 2] += (1 - scale) * center_x
            M[1, 2] += (1 - scale) * center_y

            zoomed = cv2.warpAffine(img, M, (cols, rows))
            result = cv2.addWeighted(result, 0.9, zoomed.astype(np.float32), 0.1, 0)

        return result.astype(np.uint8)

    def tilt_shift(self, img, **params):
        """افکت Tilt-Shift"""
        focus_height = params.get('focus_height', 0.3)

        rows, cols = img.shape[:2]

        # ایجاد ماسک برای ناحیه فوکوس
        mask = np.zeros((rows, cols), dtype=np.float32)
        center = rows // 2
        focus_size = int(rows * focus_height)

        mask[center - focus_size // 2:center + focus_size // 2, :] = 1

        # اعمال gradient به ماسک
        for i in range(focus_size // 2):
            fade = i / (focus_size // 2)
            mask[center - focus_size // 2 - i, :] = fade
            mask[center + focus_size // 2 + i, :] = fade

        # محو کردن قسمت‌های خارج از فوکوس
        blurred = cv2.GaussianBlur(img, (21, 21), 0)

        # ترکیب تصویر اصلی و محو شده
        result = np.zeros_like(img)
        for c in range(3):
            result[:, :, c] = img[:, :, c] * mask + blurred[:, :, c] * (1 - mask)

        # افزایش اشباع رنگ
        hsv = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.3, 0, 255)
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        return result

    def lens_blur(self, img, **params):
        """محو لنز (Bokeh)"""
        radius = params.get('radius', 10)

        # ایجاد kernel دایره‌ای برای bokeh
        kernel_size = 2 * radius + 1
        kernel = np.zeros((kernel_size, kernel_size))
        center = radius

        for i in range(kernel_size):
            for j in range(kernel_size):
                if np.sqrt((i - center) ** 2 + (j - center) ** 2) <= radius:
                    kernel[i, j] = 1

        kernel = kernel / np.sum(kernel)

        return cv2.filter2D(img, -1, kernel)

    def surface_blur(self, img, **params):
        """محو سطحی (حفظ لبه‌ها)"""
        radius = params.get('radius', 9)
        threshold = params.get('threshold', 50)

        return cv2.bilateralFilter(img, radius, threshold, threshold)

    # ==================== فیلترهای تشخیص لبه ====================

    def sobel(self, img, **params):
        """تشخیص لبه سوبل"""
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel = np.sqrt(sobelx ** 2 + sobely ** 2)
        sobel = np.uint8(np.clip(sobel, 0, 255))
        return cv2.cvtColor(sobel, cv2.COLOR_GRAY2RGB)

    def canny(self, img, **params):
        """تشخیص لبه کنی"""
        threshold1 = params.get('threshold1', 100)
        threshold2 = params.get('threshold2', 200)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, threshold1, threshold2)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

    def laplacian(self, img, **params):
        """تشخیص لبه لاپلاسین"""
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian = np.uint8(np.absolute(laplacian))
        return cv2.cvtColor(laplacian, cv2.COLOR_GRAY2RGB)

    def prewitt(self, img, **params):
        """تشخیص لبه Prewitt"""
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
        kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])

        prewittx = cv2.filter2D(gray, -1, kernelx)
        prewitty = cv2.filter2D(gray, -1, kernely)

        prewitt = np.sqrt(prewittx ** 2 + prewitty ** 2)
        prewitt = np.uint8(np.clip(prewitt, 0, 255))

        return cv2.cvtColor(prewitt, cv2.COLOR_GRAY2RGB)

    def roberts(self, img, **params):
        """تشخیص لبه Roberts"""
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        kernelx = np.array([[1, 0], [0, -1]])
        kernely = np.array([[0, 1], [-1, 0]])

        robertsx = cv2.filter2D(gray, -1, kernelx)
        robertsy = cv2.filter2D(gray, -1, kernely)

        roberts = np.sqrt(robertsx ** 2 + robertsy ** 2)
        roberts = np.uint8(np.clip(roberts, 0, 255))

        return cv2.cvtColor(roberts, cv2.COLOR_GRAY2RGB)

    # ==================== فیلترهای هنری ====================

    def emboss(self, img, **params):
        """افکت برجسته"""
        kernel = np.array([[-2, -1, 0],
                           [-1, 1, 1],
                           [0, 1, 2]])
        return cv2.filter2D(img, -1, kernel)

    def oil_painting(self, img, **params):
        """افکت نقاشی رنگ روغن - پیاده‌سازی جایگزین"""
        size = params.get('size', 7)

        # استفاده از PIL برای شبیه‌سازی افکت نقاشی رنگ روغن
        pil_img = Image.fromarray(img)

        # 1. کاهش جزئیات با فیلتر EDGE_ENHANCE_MORE
        pil_img = pil_img.filter(ImageFilter.EDGE_ENHANCE_MORE)

        # 2. اعمال Median Filter برای صاف کردن
        img_array = np.array(pil_img)
        img_array = cv2.medianBlur(img_array, size)

        # 3. افزایش اشباع رنگ
        pil_img = Image.fromarray(img_array)
        enhancer = ImageEnhance.Color(pil_img)
        pil_img = enhancer.enhance(1.4)

        # 4. کاهش تعداد رنگ‌ها با quantization
        img_array = np.array(pil_img)
        img_array = img_array // 20 * 20  # کاهش رنگ‌ها

        return img_array

    def pencil_sketch(self, img, **params):
        """افکت طراحی با مداد"""
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        inv_gray = 255 - gray
        blur = cv2.GaussianBlur(inv_gray, (21, 21), 0)
        inv_blur = 255 - blur
        sketch = cv2.divide(gray, inv_blur, scale=256.0)
        return cv2.cvtColor(sketch, cv2.COLOR_GRAY2RGB)

    def colored_pencil(self, img, **params):
        """طراحی با مداد رنگی"""
        # ایجاد sketch
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        inv_gray = 255 - gray
        blur = cv2.GaussianBlur(inv_gray, (21, 21), 0)
        inv_blur = 255 - blur
        sketch = cv2.divide(gray, inv_blur, scale=256.0)

        # ترکیب با رنگ اصلی
        sketch_colored = cv2.cvtColor(sketch, cv2.COLOR_GRAY2RGB)

        # کاهش اشباع رنگ اصلی
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] = hsv[:, :, 1] * 0.5
        desaturated = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        # ترکیب
        result = cv2.addWeighted(sketch_colored, 0.5, desaturated, 0.5, 0)

        return result

    def cartoon(self, img, **params):
        """افکت کارتونی"""
        # تبدیل به grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = cv2.medianBlur(gray, 5)

        # تشخیص لبه‌ها
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                      cv2.THRESH_BINARY, 9, 9)

        # کاهش رنگ‌ها
        data = np.float32(img).reshape((-1, 3))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)
        _, labels, centers = cv2.kmeans(data, 8, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        res = centers[labels.flatten()]
        res = res.reshape(img.shape)

        # ترکیب با لبه‌ها
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        edges = cv2.bitwise_not(edges)
        cartoon = cv2.bitwise_and(res, edges)

        return cartoon

    def watercolor(self, img, **params):
        """افکت آبرنگ"""
        # stylization برای افکت آبرنگ
        stylized = cv2.stylization(img, sigma_s=60, sigma_r=0.4)

        # اضافه کردن texture
        texture = np.random.normal(0, 5, img.shape)
        watercolor = stylized + texture

        return np.clip(watercolor, 0, 255).astype(np.uint8)

    def pointillism(self, img, **params):
        """افکت نقطه‌چینی"""
        dot_size = params.get('dot_size', 5)

        rows, cols = img.shape[:2]
        result = np.ones_like(img) * 255  # پس‌زمینه سفید

        # ایجاد نقاط
        for y in range(0, rows, dot_size * 2):
            for x in range(0, cols, dot_size * 2):
                # گرفتن رنگ میانگین منطقه
                region = img[y:y + dot_size, x:x + dot_size]
                if region.size > 0:
                    color = np.mean(region, axis=(0, 1))
                    # رسم دایره
                    cv2.circle(result, (x + dot_size // 2, y + dot_size // 2),
                               dot_size // 2, color, -1)

        return result

    def impressionist(self, img, **params):
        """افکت امپرسیونیستی"""
        brush_size = params.get('brush_size', 10)

        # اعمال فیلترهای مختلف
        result = img.copy()

        # Brush strokes simulation
        for i in range(5):
            angle = np.random.randint(0, 360)
            M = cv2.getRotationMatrix2D((brush_size // 2, brush_size // 2), angle, 1)

            kernel = np.zeros((brush_size, brush_size))
            cv2.ellipse(kernel, (brush_size // 2, brush_size // 2),
                        (brush_size // 2, brush_size // 4), angle, 0, 360, 1, -1)
            kernel = kernel / np.sum(kernel)

            result = cv2.filter2D(result, -1, kernel)

        # افزایش اشباع رنگ
        hsv = cv2.cvtColor(result, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.5, 0, 255)
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        return result

    def pop_art(self, img, **params):
        """افکت پاپ آرت"""
        # کاهش شدید رنگ‌ها
        levels = params.get('levels', 4)

        # Posterize
        indices = np.arange(0, 256)
        divider = np.linspace(0, 255, levels + 1)[1]
        quantiz = np.int0(np.linspace(0, 255, levels))
        color_levels = np.clip(np.int0(indices / divider), 0, levels - 1)

        palette = quantiz[color_levels]

        # اعمال به هر کانال
        result = np.zeros_like(img)
        for i in range(3):
            result[:, :, i] = palette[img[:, :, i]]

        # افزایش کنتراست و اشباع
        result = cv2.convertScaleAbs(result, alpha=1.5, beta=0)

        # اضافه کردن outline
        gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

        # ترکیب
        result[edges > 0] = [0, 0, 0]

        return result

    def comic_book(self, img, **params):
        """افکت کتاب کمیک"""
        # تشخیص لبه برای خطوط سیاه
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                      cv2.THRESH_BINARY, 7, 7)

        # Posterization برای رنگ‌های flat
        img_posterized = self.pop_art(img, {'levels': 6})

        # ترکیب
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        edges = cv2.bitwise_not(edges)
        comic = cv2.bitwise_and(img_posterized, edges)

        # اضافه کردن halftone dots در نواحی سایه
        shadows = gray < 100
        dots = np.random.random(gray.shape) > 0.9
        halftone = shadows & dots

        for i in range(3):
            comic[:, :, i][halftone] = 0

        return comic

    def mosaic(self, img, **params):
        """افکت موزاییک"""
        block_size = params.get('block_size', 10)

        rows, cols = img.shape[:2]
        result = np.zeros_like(img)

        for y in range(0, rows, block_size):
            for x in range(0, cols, block_size):
                # گرفتن رنگ میانگین بلوک
                block = img[y:min(y + block_size, rows), x:min(x + block_size, cols)]
                if block.size > 0:
                    avg_color = np.mean(block, axis=(0, 1))
                    result[y:min(y + block_size, rows), x:min(x + block_size, cols)] = avg_color

        return result.astype(np.uint8)

    def stained_glass(self, img, **params):
        """افکت شیشه رنگی"""
        # استفاده از superpixels
        segments = params.get('segments', 100)

        # تبدیل به LAB برای segmentation بهتر
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

        # ایجاد segments با watershed
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Noise removal
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

        # Sure background area
        sure_bg = cv2.dilate(opening, kernel, iterations=3)

        # Finding sure foreground area
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)

        # Marker labelling
        ret, markers = cv2.connectedComponents(sure_fg)

        # Add one to all labels so that sure background is not 0, but 1
        markers = markers + 1

        # Now, mark the region of unknown with zero
        markers[unknown == 255] = 0

        # Apply watershed
        markers = cv2.watershed(img, markers)

        # ایجاد تصویر نهایی
        result = img.copy()
        result[markers == -1] = [0, 0, 0]  # خطوط سیاه

        # رنگ کردن هر segment
        for label in np.unique(markers):
            if label <= 0:
                continue
            mask = markers == label
            avg_color = np.mean(img[mask], axis=0)
            result[mask] = avg_color

        return result

    # ==================== فیلترهای قدیمی و فیلم ====================

    def vintage_film(self, img, **params):
        """افکت فیلم قدیمی"""
        # کاهش کیفیت
        vintage = cv2.resize(img, None, fx=0.8, fy=0.8)
        vintage = cv2.resize(vintage, (img.shape[1], img.shape[0]))

        # اضافه کردن grain
        noise = np.random.normal(0, 15, img.shape)
        vintage = vintage + noise

        # رنگ قدیمی
        vintage_matrix = np.array([[0.5, 0.4, 0.1],
                                   [0.3, 0.5, 0.2],
                                   [0.2, 0.3, 0.5]])
        vintage = cv2.transform(vintage, vintage_matrix)

        # vignette
        rows, cols = img.shape[:2]
        kernel_x = cv2.getGaussianKernel(cols, cols / 2)
        kernel_y = cv2.getGaussianKernel(rows, rows / 2)
        kernel = kernel_y * kernel_x.T
        mask = kernel / kernel.max()

        for i in range(3):
            vintage[:, :, i] = vintage[:, :, i] * mask * 0.8

        return np.clip(vintage, 0, 255).astype(np.uint8)

    def kodachrome(self, img, **params):
        """افکت Kodachrome"""
        # ماتریس رنگ Kodachrome
        kodachrome_matrix = np.array([[1.12, -0.08, -0.04],
                                      [-0.16, 1.20, -0.04],
                                      [-0.16, -0.16, 1.32]])

        kodachrome = cv2.transform(img, kodachrome_matrix)

        # افزایش کنتراست
        kodachrome = cv2.convertScaleAbs(kodachrome, alpha=1.2, beta=10)

        return np.clip(kodachrome, 0, 255).astype(np.uint8)

    def polaroid(self, img, **params):
        """افکت پولاروید"""
        # افزایش روشنایی و کاهش کنتراست
        polaroid = cv2.convertScaleAbs(img, alpha=0.8, beta=30)

        # Shift رنگ‌ها
        polaroid[:, :, 0] = np.clip(polaroid[:, :, 0] * 1.1, 0, 255)  # Red
        polaroid[:, :, 1] = np.clip(polaroid[:, :, 1] * 1.05, 0, 255)  # Green

        # اضافه کردن حاشیه سفید
        border_size = 20
        polaroid = cv2.copyMakeBorder(polaroid, border_size, border_size * 3,
                                      border_size, border_size,
                                      cv2.BORDER_CONSTANT, value=[245, 240, 235])

        return polaroid

    def lomo(self, img, **params):
        """افکت Lomography"""
        # vignette شدید
        rows, cols = img.shape[:2]
        X_resultant_kernel = cv2.getGaussianKernel(cols, cols / 1.5)
        Y_resultant_kernel = cv2.getGaussianKernel(rows, rows / 1.5)
        kernel = Y_resultant_kernel * X_resultant_kernel.T
        mask = kernel / kernel.max()

        lomo = img.copy()
        for i in range(3):
            lomo[:, :, i] = lomo[:, :, i] * mask

        # افزایش اشباع و کنتراست
        lomo = cv2.convertScaleAbs(lomo, alpha=1.5, beta=0)

        # Color shift
        lomo[:, :, 0] = np.clip(lomo[:, :, 0] * 1.2, 0, 255)  # Red
        lomo[:, :, 2] = np.clip(lomo[:, :, 2] * 0.8, 0, 255)  # Blue

        return lomo.astype(np.uint8)

    def cross_process(self, img, **params):
        """افکت Cross Processing"""
        # تبدیل به LAB
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)

        # Shift رنگ‌ها
        lab[:, :, 1] = np.clip(lab[:, :, 1] * 1.2 - 10, -128, 127)  # a channel
        lab[:, :, 2] = np.clip(lab[:, :, 2] * 0.8 + 20, -128, 127)  # b channel

        # تبدیل به RGB
        cross = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)

        # افزایش کنتراست
        cross = cv2.convertScaleAbs(cross, alpha=1.3, beta=0)

        return cross

    def faded_film(self, img, **params):
        """افکت فیلم رنگ پریده"""
        # کاهش اشباع
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] = hsv[:, :, 1] * 0.6  # کاهش saturation
        faded = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        # افزودن رنگ زرد/نارنجی
        color_overlay = np.ones_like(faded) * [255, 200, 150]
        faded = cv2.addWeighted(faded, 0.8, color_overlay, 0.2, 0)

        # کاهش کنتراست
        faded = cv2.convertScaleAbs(faded, alpha=0.8, beta=20)

        return faded.astype(np.uint8)

    def old_photo(self, img, **params):
        """افکت عکس قدیمی"""
        # تبدیل به sepia
        old = self.sepia(img)

        # اضافه کردن scratches
        rows, cols = old.shape[:2]
        scratches = np.zeros((rows, cols), dtype=np.uint8)

        for _ in range(50):
            x1, y1 = np.random.randint(0, cols), np.random.randint(0, rows)
            x2, y2 = np.random.randint(0, cols), np.random.randint(0, rows)
            cv2.line(scratches, (x1, y1), (x2, y2), 255, 1)

        scratches = cv2.GaussianBlur(scratches, (3, 3), 0)
        scratches = cv2.cvtColor(scratches, cv2.COLOR_GRAY2RGB)

        # اضافه کردن dust و spots
        dust = np.random.random((rows, cols, 3)) * 50

        # ترکیب
        old = old - scratches * 0.3 - dust

        # vignette
        kernel_x = cv2.getGaussianKernel(cols, cols / 2)
        kernel_y = cv2.getGaussianKernel(rows, rows / 2)
        kernel = kernel_y * kernel_x.T
        mask = kernel / kernel.max()

        for i in range(3):
            old[:, :, i] = old[:, :, i] * mask * 0.7

        return np.clip(old, 0, 255).astype(np.uint8)

    def daguerreotype(self, img, **params):
        """افکت داگرئوتایپ (اولین تکنیک عکاسی)"""
        # تبدیل به سیاه و سفید
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # اعمال blur برای نرمی
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # افزایش کنتراست
        dag = cv2.convertScaleAbs(blurred, alpha=1.5, beta=0)

        # اضافه کردن metallic tint
        dag_colored = cv2.cvtColor(dag, cv2.COLOR_GRAY2RGB)
        dag_colored[:, :, 0] = np.clip(dag_colored[:, :, 0] * 1.1, 0, 255)
        dag_colored[:, :, 1] = np.clip(dag_colored[:, :, 1] * 1.05, 0, 255)
        dag_colored[:, :, 2] = np.clip(dag_colored[:, :, 2] * 0.9, 0, 255)

        # Oval vignette
        rows, cols = dag_colored.shape[:2]
        mask = np.zeros((rows, cols), dtype=np.float32)
        cv2.ellipse(mask, (cols // 2, rows // 2), (int(cols * 0.4), int(rows * 0.4)),
                    0, 0, 360, 1, -1)
        mask = cv2.GaussianBlur(mask, (51, 51), 0)

        for i in range(3):
            dag_colored[:, :, i] = dag_colored[:, :, i] * mask

        return dag_colored.astype(np.uint8)

    # ==================== فیلترهای خاص ====================

    def hdr(self, img, **params):
        """افکت HDR"""
        # تقسیم تصویر به نواحی روشن و تاریک
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # ایجاد چند exposure
        exposures = []
        for gamma in [0.5, 1.0, 2.0]:
            exposure = np.power(img / 255.0, gamma) * 255
            exposures.append(exposure.astype(np.uint8))

        # Tone mapping ساده
        hdr = np.zeros_like(img, dtype=np.float32)
        weights = [0.3, 0.4, 0.3]

        for i, (exp, weight) in enumerate(zip(exposures, weights)):
            hdr += exp * weight

        # Local contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

        hdr_result = hdr.astype(np.uint8)
        for i in range(3):
            hdr_result[:, :, i] = clahe.apply(hdr_result[:, :, i])

        return hdr_result

    def glamour(self, img, **params):
        """افکت گلامور"""
        # Soft focus
        blurred = cv2.GaussianBlur(img, (25, 25), 0)

        # High key lighting
        glamour = cv2.addWeighted(img, 0.7, blurred, 0.3, 20)

        # Skin smoothing effect
        glamour = cv2.bilateralFilter(glamour, 15, 80, 80)

        # افزایش روشنایی
        glamour = cv2.convertScaleAbs(glamour, alpha=1.1, beta=20)

        # Soft glow
        glow = cv2.GaussianBlur(glamour, (51, 51), 0)
        glamour = cv2.addWeighted(glamour, 0.8, glow, 0.2, 0)

        return glamour

    def dramatic(self, img, **params):
        """افکت دراماتیک"""
        # افزایش کنتراست شدید
        dramatic = cv2.convertScaleAbs(img, alpha=2.0, beta=-50)

        # Clarity enhancement
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]]) * 0.5
        dramatic = cv2.filter2D(dramatic, -1, kernel)

        # کاهش روشنایی در حاشیه‌ها
        rows, cols = dramatic.shape[:2]
        kernel_x = cv2.getGaussianKernel(cols, cols / 3)
        kernel_y = cv2.getGaussianKernel(rows, rows / 3)
        kernel = kernel_y * kernel_x.T
        mask = kernel / kernel.max()

        for i in range(3):
            dramatic[:, :, i] = dramatic[:, :, i] * mask

        return dramatic

    def dreamy(self, img, **params):
        """افکت رویایی"""
        # Soft focus با glow
        dreamy = cv2.GaussianBlur(img, (21, 21), 0)

        # Orton effect
        dreamy = cv2.addWeighted(img, 0.5, dreamy, 0.5, 0)

        # افزایش روشنایی و کاهش کنتراست
        dreamy = cv2.convertScaleAbs(dreamy, alpha=0.9, beta=30)

        # اضافه کردن رنگ صورتی/بنفش ملایم
        color_overlay = np.ones_like(dreamy) * [255, 230, 250]
        dreamy = cv2.addWeighted(dreamy, 0.9, color_overlay, 0.1, 0)

        return dreamy

    def ethereal(self, img, **params):
        """افکت اثیری"""
        # افزایش exposure
        ethereal = cv2.convertScaleAbs(img, alpha=1.3, beta=30)

        # Soft glow قوی
        glow = cv2.GaussianBlur(ethereal, (101, 101), 0)
        ethereal = cv2.addWeighted(ethereal, 0.6, glow, 0.4, 0)

        # کاهش اشباع
        hsv = cv2.cvtColor(ethereal, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] = hsv[:, :, 1] * 0.6
        ethereal = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        return ethereal

    def grunge(self, img, **params):
        """افکت گرانج"""
        # افزایش کنتراست و grain
        grunge = cv2.convertScaleAbs(img, alpha=1.5, beta=-20)

        # اضافه کردن نویز شدید
        noise = np.random.normal(0, 30, img.shape)
        grunge = grunge + noise

        # Texture overlay
        texture = np.random.random(img.shape) * 100
        grunge = cv2.addWeighted(grunge, 0.8, texture, 0.2, 0)

        # رنگ‌های تیره و کثیف
        grunge[:, :, 2] = grunge[:, :, 2] * 0.8  # کاهش آبی

        return np.clip(grunge, 0, 255).astype(np.uint8)

    def rainbow(self, img, **params):
        """افکت رنگین کمان"""
        rows, cols = img.shape[:2]

        # ایجاد gradient رنگین کمان
        rainbow = np.zeros_like(img)

        for i in range(cols):
            hue = int(180 * i / cols)
            color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2RGB)[0][0]
            rainbow[:, i] = color

        # ترکیب با تصویر اصلی
        result = cv2.addWeighted(img, 0.7, rainbow, 0.3, 0)

        return result

    def thermal(self, img, **params):
        """افکت حرارتی"""
        # تبدیل به grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # اعمال colormap حرارتی
        thermal = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
        thermal = cv2.cvtColor(thermal, cv2.COLOR_BGR2RGB)

        return thermal

    def xray(self, img, **params):
        """افکت اشعه ایکس"""
        # معکوس کردن
        xray = 255 - img

        # تبدیل به grayscale
        gray = cv2.cvtColor(xray, cv2.COLOR_RGB2GRAY)

        # افزایش کنتراست
        gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=0)

        # اضافه کردن رنگ آبی/سبز
        xray = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        xray[:, :, 1] = np.clip(xray[:, :, 1] * 1.2, 0, 255)  # Green
        xray[:, :, 2] = np.clip(xray[:, :, 2] * 1.4, 0, 255)  # Blue

        return xray

    def matrix(self, img, **params):
        """افکت ماتریکس"""
        # تبدیل به سبز دیجیتال
        matrix = img.copy()
        matrix[:, :, 0] = 0  # حذف قرمز
        matrix[:, :, 2] = matrix[:, :, 2] * 0.3  # کاهش آبی
        matrix[:, :, 1] = np.clip(matrix[:, :, 1] * 1.5, 0, 255)  # تقویت سبز

        # اضافه کردن scan lines
        rows, cols = matrix.shape[:2]
        for i in range(0, rows, 2):
            matrix[i, :] = matrix[i, :] * 0.8

        # Digital noise
        noise = np.random.randint(0, 50, size=matrix.shape)
        matrix = cv2.add(matrix, noise)

        return np.clip(matrix, 0, 255).astype(np.uint8)

    # ==================== فیلترهای تکنیکی ====================

    def sharpen(self, img, **params):
        """تیز کردن تصویر"""
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        return cv2.filter2D(img, -1, kernel)

    def denoise(self, img, **params):
        """حذف نویز"""
        h = params.get('h', 10)
        return cv2.fastNlMeansDenoisingColored(img, None, h, h, 7, 21)

    def histogram_equalization(self, img, **params):
        """یکسان‌سازی هیستوگرام"""
        # تبدیل به YCrCb
        ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)

        # اعمال equalization روی کانال Y
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])

        # تبدیل به RGB
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)

    def clahe(self, img, **params):
        """CLAHE (Contrast Limited Adaptive Histogram Equalization)"""
        clipLimit = params.get('clipLimit', 2.0)
        tileGridSize = params.get('tileGridSize', 8)

        # تبدیل به LAB
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

        # اعمال CLAHE به کانال L
        clahe = cv2.createCLAHE(clipLimit=clipLimit,
                                tileGridSize=(tileGridSize, tileGridSize))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])

        # تبدیل به RGB
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    def unsharp_mask(self, img, **params):
        """Unsharp Mask"""
        radius = params.get('radius', 5)
        amount = params.get('amount', 1.5)

        # ایجاد blur
        blurred = cv2.GaussianBlur(img, (radius * 2 + 1, radius * 2 + 1), 0)

        # محاسبه mask
        mask = cv2.subtract(img, blurred)

        # اعمال unsharp mask
        sharpened = cv2.addWeighted(img, 1.0, mask, amount, 0)

        return np.clip(sharpened, 0, 255).astype(np.uint8)

    def edge_preserve_smooth(self, img, **params):
        """صاف کردن با حفظ لبه‌ها"""
        flags = params.get('flags', cv2.RECURS_FILTER)
        sigma_s = params.get('sigma_s', 10)
        sigma_r = params.get('sigma_r', 0.15)

        return cv2.edgePreservingFilter(img, flags=flags,
                                        sigma_s=sigma_s, sigma_r=sigma_r)

    # ==================== فیلترهای تغییر شکل و دیستورشن ====================

    def fisheye(self, img, **params):
        """افکت چشم ماهی"""
        rows, cols = img.shape[:2]

        # مرکز تصویر
        cx, cy = cols // 2, rows // 2

        # شعاع
        radius = min(cx, cy)

        # ایجاد map
        map_x = np.zeros((rows, cols), np.float32)
        map_y = np.zeros((rows, cols), np.float32)

        for i in range(rows):
            for j in range(cols):
                # فاصله از مرکز
                dx = j - cx
                dy = i - cy
                r = np.sqrt(dx * dx + dy * dy)

                if r == 0:
                    map_x[i, j] = j
                    map_y[i, j] = i
                else:
                    # محاسبه distortion
                    theta = np.arctan2(dy, dx)

                    # Barrel distortion formula
                    r_dist = r * (1 + 0.3 * (r / radius) ** 2)

                    if r_dist < radius * 1.5:
                        map_x[i, j] = cx + r_dist * np.cos(theta)
                        map_y[i, j] = cy + r_dist * np.sin(theta)
                    else:
                        map_x[i, j] = j
                        map_y[i, j] = i

        return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)

    def barrel_distortion(self, img, **params):
        """دیستورشن بشکه‌ای"""
        k = params.get('k', 0.00005)

        rows, cols = img.shape[:2]
        cx, cy = cols // 2, rows // 2

        map_x = np.zeros((rows, cols), np.float32)
        map_y = np.zeros((rows, cols), np.float32)

        for i in range(rows):
            for j in range(cols):
                dx = j - cx
                dy = i - cy
                r2 = dx * dx + dy * dy

                # Barrel distortion
                factor = 1 + k * r2

                map_x[i, j] = cx + dx * factor
                map_y[i, j] = cy + dy * factor

        return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)

    def pincushion(self, img, **params):
        """دیستورشن بالشتکی"""
        k = params.get('k', -0.00005)

        rows, cols = img.shape[:2]
        cx, cy = cols // 2, rows // 2

        map_x = np.zeros((rows, cols), np.float32)
        map_y = np.zeros((rows, cols), np.float32)

        for i in range(rows):
            for j in range(cols):
                dx = j - cx
                dy = i - cy
                r2 = dx * dx + dy * dy

                # Pincushion distortion
                factor = 1 + k * r2

                map_x[i, j] = cx + dx * factor
                map_y[i, j] = cy + dy * factor

        return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)

    def wave_distortion(self, img, **params):
        """دیستورشن موجی"""
        amplitude = params.get('amplitude', 20)
        frequency = params.get('frequency', 0.05)

        rows, cols = img.shape[:2]

        map_x = np.zeros((rows, cols), np.float32)
        map_y = np.zeros((rows, cols), np.float32)

        for i in range(rows):
            for j in range(cols):
                map_x[i, j] = j + amplitude * np.sin(2 * np.pi * i * frequency)
                map_y[i, j] = i

        return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)

    def swirl(self, img, **params):
        """افکت چرخش"""
        strength = params.get('strength', 0.5)

        rows, cols = img.shape[:2]
        cx, cy = cols // 2, rows // 2

        map_x = np.zeros((rows, cols), np.float32)
        map_y = np.zeros((rows, cols), np.float32)

        max_radius = np.sqrt(cx ** 2 + cy ** 2)

        for i in range(rows):
            for j in range(cols):
                dx = j - cx
                dy = i - cy
                r = np.sqrt(dx * dx + dy * dy)

                if r > 0:
                    theta = np.arctan2(dy, dx)
                    swirl_amount = 1.0 - (r / max_radius)
                    swirl_angle = strength * swirl_amount * np.pi * 2

                    new_theta = theta + swirl_angle

                    map_x[i, j] = cx + r * np.cos(new_theta)
                    map_y[i, j] = cy + r * np.sin(new_theta)
                else:
                    map_x[i, j] = j
                    map_y[i, j] = i

        return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)

    def pixelate(self, img, **params):
        """پیکسلی کردن"""
        pixel_size = params.get('pixel_size', 10)

        # کوچک کردن و بزرگ کردن دوباره
        height, width = img.shape[:2]

        # کوچک کردن
        small = cv2.resize(img, (width // pixel_size, height // pixel_size),
                           interpolation=cv2.INTER_NEAREST)

        # بزرگ کردن با interpolation nearest
        pixelated = cv2.resize(small, (width, height),
                               interpolation=cv2.INTER_NEAREST)

        return pixelated

    def crystallize(self, img, **params):
        """افکت کریستالی"""
        size = params.get('size', 20)

        rows, cols = img.shape[:2]
        result = np.zeros_like(img)

        # ایجاد نقاط تصادفی برای مراکز کریستال
        num_crystals = (rows * cols) // (size * size)
        centers = []

        for _ in range(num_crystals):
            x = np.random.randint(0, cols)
            y = np.random.randint(0, rows)
            centers.append((x, y))

        # برای هر پیکسل، پیدا کردن نزدیکترین مرکز
        for i in range(rows):
            for j in range(cols):
                min_dist = float('inf')
                closest_center = None

                for cx, cy in centers:
                    dist = np.sqrt((j - cx) ** 2 + (i - cy) ** 2)
                    if dist < min_dist:
                        min_dist = dist
                        closest_center = (cx, cy)

                if closest_center:
                    cx, cy = closest_center
                    if 0 <= cy < rows and 0 <= cx < cols:
                        result[i, j] = img[cy, cx]

        return result

    # ==================== فیلترهای نور و سایه ====================

    def vignette(self, img, **params):
        """افکت وینیت"""
        strength = params.get('strength', 0.8)

        rows, cols = img.shape[:2]

        # ایجاد ماسک گاوسی
        kernel_x = cv2.getGaussianKernel(cols, cols / 2)
        kernel_y = cv2.getGaussianKernel(rows, rows / 2)
        kernel = kernel_y * kernel_x.T
        mask = kernel / kernel.max()

        # تنظیم قدرت
        mask = 1 - (1 - mask) * strength

        # اعمال به تصویر
        result = img.copy()
        for i in range(3):
            result[:, :, i] = result[:, :, i] * mask

        return result.astype(np.uint8)

    def light_leak(self, img, **params):
        """افکت نشت نور"""
        color = params.get('color', [255, 200, 100])
        position = params.get('position', 'top-right')

        rows, cols = img.shape[:2]

        # ایجاد gradient
        leak = np.zeros_like(img)

        if position == 'top-right':
            for i in range(rows):
                for j in range(cols):
                    distance = np.sqrt((j - cols) ** 2 + i ** 2)
                    intensity = np.exp(-distance / (cols * 0.5))
                    leak[i, j] = np.array(color) * intensity

        # ترکیب با تصویر
        result = cv2.addWeighted(img, 0.8, leak.astype(np.uint8), 0.2, 0)

        return result

    def lens_flare(self, img, **params):
        """افکت فلر لنز"""
        center_x = params.get('center_x', img.shape[1] // 3)
        center_y = params.get('center_y', img.shape[0] // 3)
        radius = params.get('radius', 100)

        rows, cols = img.shape[:2]

        # ایجاد فلر اصلی
        flare = np.zeros_like(img)

        for i in range(rows):
            for j in range(cols):
                distance = np.sqrt((j - center_x) ** 2 + (i - center_y) ** 2)
                if distance < radius:
                    intensity = 1 - (distance / radius)
                    flare[i, j] = [255, 255, 200] * intensity

        # اضافه کردن artifacts
        for i in range(3):
            artifact_x = center_x + (cols // 2 - center_x) * (i + 1) / 4
            artifact_y = center_y + (rows // 2 - center_y) * (i + 1) / 4
            artifact_radius = radius // (i + 2)

            cv2.circle(flare, (int(artifact_x), int(artifact_y)),
                       artifact_radius, (100, 150, 200), -1)

        # blur برای نرمی
        flare = cv2.GaussianBlur(flare, (51, 51), 0)

        # ترکیب
        result = cv2.add(img, flare // 3)

        return np.clip(result, 0, 255).astype(np.uint8)

    def sun_rays(self, img, **params):
        """افکت پرتوهای خورشید"""
        center_x = params.get('center_x', img.shape[1] // 2)
        center_y = params.get('center_y', 0)
        num_rays = params.get('num_rays', 8)

        rows, cols = img.shape[:2]
        rays = np.zeros_like(img, dtype=np.float32)

        for ray in range(num_rays):
            angle = (2 * np.pi * ray) / num_rays

            # ایجاد gradient برای هر پرتو
            for r in range(max(rows, cols)):
                x = int(center_x + r * np.cos(angle))
                y = int(center_y + r * np.sin(angle))

                if 0 <= x < cols and 0 <= y < rows:
                    intensity = np.exp(-r / 200)
                    cv2.line(rays, (center_x, center_y), (x, y),
                             (255, 255, 200), 2)

        # blur برای نرمی
        rays = cv2.GaussianBlur(rays, (61, 61), 0)

        # ترکیب
        result = cv2.addWeighted(img, 0.8, rays.astype(np.uint8), 0.2, 0)

        return result

    def soft_light(self, img, **params):
        """نور نرم"""
        # ایجاد overlay روشن
        overlay = np.ones_like(img) * 255

        # Gaussian blur
        soft = cv2.GaussianBlur(img, (51, 51), 0)

        # Soft light blending
        result = img.copy().astype(np.float32)

        for i in range(3):
            result[:, :, i] = (soft[:, :, i] / 255) * (result[:, :, i] +
                                                       2 * soft[:, :, i] * (255 - result[:, :, i]) / 255)

        return np.clip(result, 0, 255).astype(np.uint8)

    def hard_light(self, img, **params):
        """نور سخت"""
        # افزایش کنتراست شدید
        result = cv2.convertScaleAbs(img, alpha=1.8, beta=-40)

        # اضافه کردن highlights
        gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
        highlights = gray > 200

        result[highlights] = [255, 255, 255]

        return result
