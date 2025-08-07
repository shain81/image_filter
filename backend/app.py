from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
from PIL import Image
import io
import base64
from filters import ImageFilters
from realtime_filters import RealtimeFilterProcessor, REALTIME_OPTIMIZED_FILTERS, HEAVY_FILTERS
import uuid

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", max_http_buffer_size=10*1024*1024)

# تنظیمات
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# ایجاد پوشه آپلود
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ایجاد instance از فیلترها
image_filters = ImageFilters()
realtime_processor = RealtimeFilterProcessor()

def allowed_file(filename):
    """بررسی فرمت فایل"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def image_to_base64(image_array):
    """تبدیل آرایه numpy به base64"""
    pil_img = Image.fromarray(image_array.astype('uint8'))
    buffered = io.BytesIO()
    pil_img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

# WebSocket Events for Real-time filtering
@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('connected', {'status': 'Connected to real-time filter server'})

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('set_filter')
def handle_set_filter(data):
    """تنظیم فیلتر برای پردازش real-time"""
    filter_name = data.get('filter', 'original')
    params = data.get('params', {})
    
    print(f"Setting filter: {filter_name} with params: {params}")  # Debug
    
    realtime_processor.set_filter(filter_name, params)
    emit('filter_set', {'filter': filter_name, 'status': 'success'})

@socketio.on('process_frame')
def handle_process_frame(data):
    """پردازش یک فریم ویدیو"""
    try:
        frame_data = data.get('frame')
        if frame_data:
            processed_frame = realtime_processor.process_frame(frame_data)
            if processed_frame:
                emit('processed_frame', {'frame': processed_frame})
            else:
                # اگر فریم پردازش نشد، فریم اصلی را برگردان
                emit('processed_frame', {'frame': frame_data})
    except Exception as e:
        print(f"Error in handle_process_frame: {e}")
        # در صورت خطا، فریم اصلی را برگردان
        emit('processed_frame', {'frame': data.get('frame')})

@app.route('/api/realtime/filters', methods=['GET'])
def get_realtime_filters():
    """دریافت لیست فیلترهای مناسب برای real-time"""
    filters = []
    all_filters = get_filters().json['filters']
    
    for f in all_filters:
        if f['id'] in REALTIME_OPTIMIZED_FILTERS:
            f['realtime_safe'] = True
            f['performance'] = 'optimal'
            filters.append(f)
        elif f['id'] in HEAVY_FILTERS:
            f['realtime_safe'] = False
            f['performance'] = 'heavy'
            f['performance_warning'] = 'ممکن است کندی ایجاد کند'
            filters.append(f)
        else:
            # فیلترهای متوسط
            f['realtime_safe'] = True
            f['performance'] = 'medium'
            filters.append(f)
    
    return jsonify({'filters': filters})

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """آپلود تصویر"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'فایلی انتخاب نشده است'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'فایلی انتخاب نشده است'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'فرمت فایل پشتیبانی نمی‌شود'}), 400
        
        # بررسی حجم فایل
        file.seek(0, os.SEEK_END)
        file_length = file.tell()
        if file_length > MAX_FILE_SIZE:
            return jsonify({'error': 'حجم فایل بیش از 10 مگابایت است'}), 400
        file.seek(0)
        
        # ذخیره فایل با نام یکتا
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        filepath = os.path.join(UPLOAD_FOLDER, unique_filename)
        file.save(filepath)
        
        # خواندن تصویر و ارسال اطلاعات اولیه
        img = cv2.imread(filepath)
        height, width = img.shape[:2]
        
        return jsonify({
            'success': True,
            'filename': unique_filename,
            'original_name': filename,
            'width': width,
            'height': height
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/apply-filter', methods=['POST'])
def apply_filter():
    """اعمال فیلتر روی تصویر"""
    try:
        data = request.json
        filename = data.get('filename')
        filter_name = data.get('filter')
        params = data.get('params', {})
        
        if not filename or not filter_name:
            return jsonify({'error': 'پارامترهای ناقص'}), 400
        
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        
        if not os.path.exists(filepath):
            return jsonify({'error': 'فایل یافت نشد'}), 404
        
        # اعمال فیلتر
        filtered_image = image_filters.apply_filter(filepath, filter_name, **params)
        
        # تبدیل به base64
        image_base64 = image_to_base64(filtered_image)
        
        return jsonify({
            'success': True,
            'image': image_base64,
            'filter': filter_name
        })
        
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': f'خطا در اعمال فیلتر: {str(e)}'}), 500

@app.route('/api/get-filters', methods=['GET'])
def get_filters():
    """دریافت لیست فیلترهای موجود"""
    filters = [
        # فیلترهای پایه - بهینه برای real-time
        {'id': 'original', 'name': 'تصویر اصلی', 'category': 'basic'},
        {'id': 'grayscale', 'name': 'سیاه و سفید', 'category': 'basic'},
        {'id': 'sepia', 'name': 'سپیا', 'category': 'basic'},
        {'id': 'negative', 'name': 'نگاتیو', 'category': 'basic'},
        {'id': 'brightness', 'name': 'روشنایی', 'category': 'basic', 'params': [{'name': 'factor', 'min': 0.5, 'max': 2.0, 'default': 1.5}]},
        {'id': 'contrast', 'name': 'کنتراست', 'category': 'basic', 'params': [{'name': 'factor', 'min': 0.5, 'max': 2.0, 'default': 1.5}]},
        
        # فیلترهای رنگی - بهینه برای real-time
        {'id': 'warm', 'name': 'گرم', 'category': 'color'},
        {'id': 'cool', 'name': 'سرد', 'category': 'color'},
        {'id': 'vintage', 'name': 'وینتیج', 'category': 'color'},
        {'id': 'cyberpunk', 'name': 'سایبرپانک', 'category': 'color'},
        {'id': 'sunset', 'name': 'غروب', 'category': 'color'},
        {'id': 'night', 'name': 'شب', 'category': 'color'},
        {'id': 'autumn', 'name': 'پاییز', 'category': 'color'},
        {'id': 'spring', 'name': 'بهار', 'category': 'color'},
        {'id': 'purple_haze', 'name': 'مه بنفش', 'category': 'color'},
        {'id': 'golden_hour', 'name': 'ساعت طلایی', 'category': 'color'},
        {'id': 'neon', 'name': 'نئون', 'category': 'color'},
        {'id': 'pastel', 'name': 'پاستل', 'category': 'color'},
        {'id': 'duotone', 'name': 'دو رنگ', 'category': 'color'},
        {'id': 'tritone', 'name': 'سه رنگ', 'category': 'color'},
        {'id': 'hue_shift', 'name': 'تغییر رنگ', 'category': 'color', 'params': [{'name': 'shift', 'min': -90, 'max': 90, 'default': 30}]},
        {'id': 'saturation', 'name': 'اشباع رنگ', 'category': 'color', 'params': [{'name': 'factor', 'min': 0, 'max': 3, 'default': 1.5}]},
        {'id': 'vibrance', 'name': 'جذابیت رنگ', 'category': 'color', 'params': [{'name': 'amount', 'min': -100, 'max': 100, 'default': 50}]},
        {'id': 'color_balance', 'name': 'تعادل رنگ', 'category': 'color', 'params': [
            {'name': 'red', 'min': 0.5, 'max': 1.5, 'default': 1.0},
            {'name': 'green', 'min': 0.5, 'max': 1.5, 'default': 1.0},
            {'name': 'blue', 'min': 0.5, 'max': 1.5, 'default': 1.0}
        ]},
        
        # فیلترهای سیاه و سفید پیشرفته - بهینه برای real-time
        {'id': 'bw_high_contrast', 'name': 'سیاه سفید کنتراست بالا', 'category': 'blackwhite'},
        {'id': 'bw_low_contrast', 'name': 'سیاه سفید کنتراست پایین', 'category': 'blackwhite'},
        {'id': 'bw_red_filter', 'name': 'سیاه سفید فیلتر قرمز', 'category': 'blackwhite'},
        {'id': 'bw_green_filter', 'name': 'سیاه سفید فیلتر سبز', 'category': 'blackwhite'},
        {'id': 'bw_blue_filter', 'name': 'سیاه سفید فیلتر آبی', 'category': 'blackwhite'},
        {'id': 'bw_orange_filter', 'name': 'سیاه سفید فیلتر نارنجی', 'category': 'blackwhite'},
        {'id': 'bw_yellow_filter', 'name': 'سیاه سفید فیلتر زرد', 'category': 'blackwhite'},
        {'id': 'ansel_adams', 'name': 'سبک انسل آدامز', 'category': 'blackwhite'},
        {'id': 'film_noir', 'name': 'فیلم نوآر', 'category': 'blackwhite'},
        {'id': 'infrared', 'name': 'مادون قرمز', 'category': 'blackwhite'},
        
        # فیلترهای محو - برخی بهینه برای real-time
        {'id': 'gaussian_blur', 'name': 'محو گاوسی', 'category': 'blur', 'params': [{'name': 'kernel_size', 'min': 3, 'max': 31, 'default': 15, 'step': 2}]},
        {'id': 'motion_blur', 'name': 'محو حرکتی', 'category': 'blur', 'params': [
            {'name': 'size', 'min': 5, 'max': 25, 'default': 15, 'step': 2},
            {'name': 'angle', 'min': 0, 'max': 360, 'default': 0}
        ]},
        {'id': 'box_blur', 'name': 'محو جعبه‌ای', 'category': 'blur', 'params': [{'name': 'kernel_size', 'min': 3, 'max': 21, 'default': 9, 'step': 2}]},
        {'id': 'radial_blur', 'name': 'محو شعاعی', 'category': 'blur', 'params': [{'name': 'strength', 'min': 5, 'max': 20, 'default': 10}]},
        {'id': 'zoom_blur', 'name': 'محو زوم', 'category': 'blur', 'params': [{'name': 'strength', 'min': 0.1, 'max': 0.5, 'default': 0.2}]},
        {'id': 'tilt_shift', 'name': 'تیلت شیفت', 'category': 'blur', 'params': [{'name': 'focus_height', 'min': 0.1, 'max': 0.5, 'default': 0.3}]},
        {'id': 'lens_blur', 'name': 'محو لنز', 'category': 'blur', 'params': [{'name': 'radius', 'min': 5, 'max': 20, 'default': 10}]},
        {'id': 'surface_blur', 'name': 'محو سطحی', 'category': 'blur', 'params': [
            {'name': 'radius', 'min': 5, 'max': 15, 'default': 9},
            {'name': 'threshold', 'min': 10, 'max': 100, 'default': 50}
        ]},
        
        # فیلترهای تشخیص لبه
        {'id': 'sobel', 'name': 'تشخیص لبه سوبل', 'category': 'edge'},
        {'id': 'canny', 'name': 'تشخیص لبه کنی', 'category': 'edge', 'params': [
            {'name': 'threshold1', 'min': 50, 'max': 200, 'default': 100},
            {'name': 'threshold2', 'min': 100, 'max': 300, 'default': 200}
        ]},
        {'id': 'laplacian', 'name': 'تشخیص لبه لاپلاسین', 'category': 'edge'},
        {'id': 'prewitt', 'name': 'تشخیص لبه پرویت', 'category': 'edge'},
        {'id': 'roberts', 'name': 'تشخیص لبه رابرتس', 'category': 'edge'},
        
        # فیلترهای هنری - بهینه برای real-time
        {'id': 'emboss', 'name': 'برجسته', 'category': 'artistic'},
        {'id': 'oil_painting', 'name': 'نقاشی رنگ روغن', 'category': 'artistic', 'params': [{'name': 'size', 'min': 3, 'max': 15, 'default': 7, 'step': 2}]},
        {'id': 'pencil_sketch', 'name': 'طراحی با مداد', 'category': 'artistic'},
        {'id': 'colored_pencil', 'name': 'مداد رنگی', 'category': 'artistic'},
        {'id': 'cartoon', 'name': 'کارتونی', 'category': 'artistic'},
        {'id': 'watercolor', 'name': 'آبرنگ', 'category': 'artistic'},
        {'id': 'pointillism', 'name': 'نقطه‌چینی', 'category': 'artistic', 'params': [{'name': 'dot_size', 'min': 3, 'max': 10, 'default': 5}]},
        {'id': 'impressionist', 'name': 'امپرسیونیست', 'category': 'artistic', 'params': [{'name': 'brush_size', 'min': 5, 'max': 20, 'default': 10}]},
        {'id': 'pop_art', 'name': 'پاپ آرت', 'category': 'artistic', 'params': [{'name': 'levels', 'min': 2, 'max': 8, 'default': 4}]},
        {'id': 'comic_book', 'name': 'کتاب کمیک', 'category': 'artistic'},
        {'id': 'mosaic', 'name': 'موزاییک', 'category': 'artistic', 'params': [{'name': 'block_size', 'min': 5, 'max': 30, 'default': 10}]},
        {'id': 'stained_glass', 'name': 'شیشه رنگی', 'category': 'artistic', 'params': [{'name': 'segments', 'min': 50, 'max': 200, 'default': 100}]},
        
        # فیلترهای قدیمی
        {'id': 'vintage_film', 'name': 'فیلم قدیمی', 'category': 'vintage'},
        {'id': 'kodachrome', 'name': 'کداکروم', 'category': 'vintage'},
        {'id': 'polaroid', 'name': 'پولاروید', 'category': 'vintage'},
        {'id': 'lomo', 'name': 'لومو', 'category': 'vintage'},
        {'id': 'cross_process', 'name': 'کراس پروسس', 'category': 'vintage'},
        {'id': 'faded_film', 'name': 'فیلم رنگ پریده', 'category': 'vintage'},
        {'id': 'old_photo', 'name': 'عکس قدیمی', 'category': 'vintage'},
        {'id': 'daguerreotype', 'name': 'داگرئوتایپ', 'category': 'vintage'},
        
        # فیلترهای خاص - بهینه برای real-time
        {'id': 'hdr', 'name': 'HDR', 'category': 'special'},
        {'id': 'glamour', 'name': 'گلامور', 'category': 'special'},
        {'id': 'dramatic', 'name': 'دراماتیک', 'category': 'special'},
        {'id': 'dreamy', 'name': 'رویایی', 'category': 'special'},
        {'id': 'ethereal', 'name': 'اثیری', 'category': 'special'},
        {'id': 'grunge', 'name': 'گرانج', 'category': 'special'},
        {'id': 'rainbow', 'name': 'رنگین کمان', 'category': 'special'},
        {'id': 'thermal', 'name': 'حرارتی', 'category': 'special'},
        {'id': 'xray', 'name': 'اشعه ایکس', 'category': 'special'},
        {'id': 'matrix', 'name': 'ماتریکس', 'category': 'special'},
        
        # فیلترهای بهبود
        {'id': 'sharpen', 'name': 'تیز کردن', 'category': 'enhancement'},
        {'id': 'denoise', 'name': 'حذف نویز', 'category': 'enhancement', 'params': [{'name': 'h', 'min': 5, 'max': 20, 'default': 10}]},
        {'id': 'histogram_eq', 'name': 'بهبود کنتراست', 'category': 'enhancement'},
        {'id': 'clahe', 'name': 'CLAHE', 'category': 'enhancement', 'params': [
            {'name': 'clipLimit', 'min': 1.0, 'max': 5.0, 'default': 2.0},
            {'name': 'tileGridSize', 'min': 4, 'max': 16, 'default': 8}
        ]},
        {'id': 'unsharp_mask', 'name': 'Unsharp Mask', 'category': 'enhancement', 'params': [
            {'name': 'radius', 'min': 1, 'max': 10, 'default': 5},
            {'name': 'amount', 'min': 0.5, 'max': 3.0, 'default': 1.5}
        ]},
        {'id': 'edge_preserve', 'name': 'صاف با حفظ لبه', 'category': 'enhancement', 'params': [
            {'name': 'sigma_s', 'min': 5, 'max': 20, 'default': 10},
            {'name': 'sigma_r', 'min': 0.1, 'max': 0.3, 'default': 0.15}
        ]},
        
        # فیلترهای دیستورشن - بهینه برای real-time
        {'id': 'fisheye', 'name': 'چشم ماهی', 'category': 'distortion'},
        {'id': 'barrel_distortion', 'name': 'دیستورشن بشکه‌ای', 'category': 'distortion', 'params': [{'name': 'k', 'min': 0.00001, 'max': 0.0001, 'default': 0.00005}]},
        {'id': 'pincushion', 'name': 'دیستورشن بالشتکی', 'category': 'distortion', 'params': [{'name': 'k', 'min': -0.0001, 'max': -0.00001, 'default': -0.00005}]},
        {'id': 'wave', 'name': 'موج', 'category': 'distortion', 'params': [
            {'name': 'amplitude', 'min': 10, 'max': 50, 'default': 20},
            {'name': 'frequency', 'min': 0.01, 'max': 0.1, 'default': 0.05}
        ]},
        {'id': 'swirl', 'name': 'چرخش', 'category': 'distortion', 'params': [{'name': 'strength', 'min': 0.1, 'max': 1.0, 'default': 0.5}]},
        {'id': 'pixelate', 'name': 'پیکسلی', 'category': 'distortion', 'params': [{'name': 'pixel_size', 'min': 5, 'max': 30, 'default': 10}]},
        {'id': 'crystallize', 'name': 'کریستالی', 'category': 'distortion', 'params': [{'name': 'size', 'min': 10, 'max': 50, 'default': 20}]},
        
        # فیلترهای نور - بهینه برای real-time
        {'id': 'vignette', 'name': 'وینیت', 'category': 'light', 'params': [{'name': 'strength', 'min': 0.1, 'max': 1.0, 'default': 0.8}]},
        {'id': 'light_leak', 'name': 'نشت نور', 'category': 'light'},
        {'id': 'lens_flare', 'name': 'فلر لنز', 'category': 'light', 'params': [
            {'name': 'center_x', 'min': 0, 'max': 100, 'default': 33},
            {'name': 'center_y', 'min': 0, 'max': 100, 'default': 33},
            {'name': 'radius', 'min': 50, 'max': 200, 'default': 100}
        ]},
        {'id': 'sun_rays', 'name': 'پرتوهای خورشید', 'category': 'light', 'params': [
            {'name': 'center_x', 'min': 0, 'max': 100, 'default': 50},
            {'name': 'center_y', 'min': 0, 'max': 100, 'default': 0},
            {'name': 'num_rays', 'min': 4, 'max': 16, 'default': 8}
        ]},
        {'id': 'soft_light', 'name': 'نور نرم', 'category': 'light'},
        {'id': 'hard_light', 'name': 'نور سخت', 'category': 'light'},
    ]
    
    return jsonify({'filters': filters})

@app.route('/api/download/<filter_name>/<filename>', methods=['GET'])
def download_image(filter_name, filename):
    """دانلود تصویر فیلتر شده"""
    try:
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        
        if not os.path.exists(filepath):
            return jsonify({'error': 'فایل یافت نشد'}), 404
        
        # اعمال فیلتر
        filtered_image = image_filters.apply_filter(filepath, filter_name)
        
        # ذخیره در حافظه
        pil_img = Image.fromarray(filtered_image.astype('uint8'))
        img_io = io.BytesIO()
        pil_img.save(img_io, 'PNG')
        img_io.seek(0)
        
        return send_file(img_io, mimetype='image/png', 
                        download_name=f'{filter_name}_{filename}')
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/cleanup/<filename>', methods=['DELETE'])
def cleanup(filename):
    """حذف فایل آپلود شده"""
    try:
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    socketio.run(app, debug=True, port=5000, host='0.0.0.0')