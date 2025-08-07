// متغیرهای سراسری
let socket;
let stream;
let videoElement;
let canvasOriginal, canvasFiltered;
let ctxOriginal, ctxFiltered;
let currentFilter = 'original';
let isProcessing = false;
let frameCount = 0;
let lastTime = Date.now();
let frameSkip = 0;
let processStartTime = 0;

// اتصال به سرور
function connectSocket() {
    socket = io('http://localhost:5000');
    
    socket.on('connect', function() {
        console.log('Connected to server');
        updateConnectionStatus('متصل');
    });

    socket.on('disconnect', function() {
        console.log('Disconnected from server');
        updateConnectionStatus('قطع شده');
    });

    socket.on('processed_frame', function(data) {
        if (data.frame) {
            displayProcessedFrame(data.frame);
            isProcessing = false;
            updateDelay();
        }
    });

    socket.on('filter_set', function(data) {
        console.log('Filter set:', data.filter);
        updateActiveFilter(data.filter);
    });
}

// شروع دوربین
async function startCamera() {
    try {
        const constraints = {
            video: {
                width: { ideal: 640 },
                height: { ideal: 480 },
                facingMode: "user"
            }
        };

        stream = await navigator.mediaDevices.getUserMedia(constraints);
        
        // ایجاد video element مخفی
        videoElement = document.createElement('video');
        videoElement.srcObject = stream;
        videoElement.play();

        // تنظیم canvas ها
        canvasOriginal = document.querySelector('.video-container:first-child canvas');
        canvasFiltered = document.querySelector('.video-container:last-child canvas');
        ctxOriginal = canvasOriginal.getContext('2d');
        ctxFiltered = canvasFiltered.getContext('2d');

        // شروع پردازش
        videoElement.onloadedmetadata = () => {
            updateResolution(videoElement.videoWidth, videoElement.videoHeight);
            requestAnimationFrame(processFrame);
        };

        // تغییر وضعیت دکمه‌ها
        document.querySelector('button:contains("شروع")').style.display = 'none';
        document.querySelector('button:contains("توقف")').style.display = 'inline-block';

    } catch (err) {
        console.error('Error accessing camera:', err);
        alert('خطا در دسترسی به دوربین: ' + err.message);
    }
}

// پردازش فریم
function processFrame() {
    if (!stream || !stream.active) return;

    // رسم تصویر اصلی
    ctxOriginal.drawImage(videoElement, 0, 0, canvasOriginal.width, canvasOriginal.height);

    // محاسبه FPS
    frameCount++;
    const currentTime = Date.now();
    if (currentTime - lastTime >= 1000) {
        updateFPS(frameCount);
        frameCount = 0;
        lastTime = currentTime;
    }

    // پردازش هر چند فریم یکبار (برای performance)
    frameSkip++;
    if (!isProcessing && socket && socket.connected && frameSkip >= 2) {
        frameSkip = 0;
        isProcessing = true;
        processStartTime = Date.now();
        
        const frameData = canvasOriginal.toDataURL('image/jpeg', 0.8);
        socket.emit('process_frame', { frame: frameData });
    }

    requestAnimationFrame(processFrame);
}

// نمایش فریم پردازش شده
function displayProcessedFrame(frameData) {
    const img = new Image();
    img.onload = function() {
        ctxFiltered.drawImage(img, 0, 0, canvasFiltered.width, canvasFiltered.height);
    };
    img.src = frameData;
}

// توقف دوربین
function stopCamera() {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }
    
    // پاک کردن canvas
    if (ctxOriginal) ctxOriginal.clearRect(0, 0, canvasOriginal.width, canvasOriginal.height);
    if (ctxFiltered) ctxFiltered.clearRect(0, 0, canvasFiltered.width, canvasFiltered.height);
    
    // تغییر دکمه‌ها
    const startBtn = Array.from(document.querySelectorAll('button')).find(btn => btn.textContent.includes('شروع'));
    const stopBtn = Array.from(document.querySelectorAll('button')).find(btn => btn.textContent.includes('توقف'));
    
    if (startBtn) startBtn.style.display = 'inline-block';
    if (stopBtn) stopBtn.style.display = 'none';
}

// تغییر دوربین
async function switchCamera() {
    if (!stream) return;
    
    const currentFacingMode = stream.getVideoTracks()[0].getSettings().facingMode;
    const newFacingMode = currentFacingMode === 'user' ? 'environment' : 'user';
    
    stopCamera();
    
    const constraints = {
        video: {
            width: { ideal: 640 },
            height: { ideal: 480 },
            facingMode: newFacingMode
        }
    };
    
    try {
        stream = await navigator.mediaDevices.getUserMedia(constraints);
        videoElement.srcObject = stream;
        
        // دوباره شروع کنیم
        const startBtn = Array.from(document.querySelectorAll('button')).find(btn => btn.textContent.includes('شروع'));
        const stopBtn = Array.from(document.querySelectorAll('button')).find(btn => btn.textContent.includes('توقف'));
        
        if (startBtn) startBtn.style.display = 'none';
        if (stopBtn) stopBtn.style.display = 'inline-block';
        
        requestAnimationFrame(processFrame);
    } catch (err) {
        console.error('Error switching camera:', err);
    }
}

// عکس گرفتن
function capturePhoto() {
    if (!canvasFiltered) return;
    
    const link = document.createElement('a');
    link.download = `photo_${currentFilter}_${Date.now()}.jpg`;
    link.href = canvasFiltered.toDataURL('image/jpeg', 0.95);
    link.click();
}

// آپدیت UI
function updateFPS(fps) {
    const elements = document.querySelectorAll('.stats-item');
    elements[0].querySelector('div:last-child').textContent = fps;
}

function updateDelay() {
    const delay = Date.now() - processStartTime;
    const elements = document.querySelectorAll('.stats-item');
    elements[3].querySelector('div:last-child').textContent = delay + 'ms';
}

function updateResolution(width, height) {
    const elements = document.querySelectorAll('.stats-item');
    elements[1].querySelector('div:last-child').textContent = `${width}x${height}`;
}

function updateActiveFilter(filterName) {
    const elements = document.querySelectorAll('.stats-item');
    elements[2].querySelector('div:last-child').textContent = filterName;
    currentFilter = filterName;
}

function updateConnectionStatus(status) {
    // اگر المان وضعیت اتصال دارید
    const statusElement = document.querySelector('.connection-status');
    if (statusElement) {
        statusElement.textContent = status;
    }
}

// Event Listeners
document.addEventListener('DOMContentLoaded', function() {
    connectSocket();

    // دکمه‌های کنترل
    const buttons = document.querySelectorAll('button');
    buttons.forEach(btn => {
        if (btn.textContent.includes('شروع')) {
            btn.addEventListener('click', startCamera);
        } else if (btn.textContent.includes('توقف')) {
            btn.addEventListener('click', stopCamera);
        } else if (btn.textContent.includes('تغییر')) {
            btn.addEventListener('click', switchCamera);
        } else if (btn.textContent.includes('عکس')) {
            btn.addEventListener('click', capturePhoto);
        }
    });

    // فیلترها
    const filterButtons = document.querySelectorAll('.filter-grid button');
    filterButtons.forEach(btn => {
        btn.addEventListener('click', function() {
            // نام فیلتر را از متن دکمه بگیرید
            let filterName = this.textContent.trim();
            
            // تبدیل نام فارسی به انگلیسی
            const filterMap = {
                'اصلی': 'original',
                'سیاه و سفید': 'grayscale',
                'سپیا': 'sepia',
                'نگاتیو': 'negative',
                'گرم': 'warm',
                'سرد': 'cool',
                'وینتیج': 'vintage',
                'سایبرپانک': 'cyberpunk',
                'غروب': 'sunset',
                'شب': 'night',
                'پاییز': 'autumn',
                'بهار': 'spring',
                'مه بنفش': 'purple_haze',
                'ساعت طلایی': 'golden_hour',
                'نئون': 'neon',
                'پاستل': 'pastel',
                'محو گاوسی': 'gaussian_blur',
                'تشخیص لبه سوبل': 'sobel',
                'تشخیص لبه کنی': 'canny',
                'اثر نقاشی': 'oil_painting',
                'کارتونی': 'cartoon',
                'طراحی با مداد': 'pencil_sketch',
                'آبرنگ': 'watercolor',
                'پاپ آرت': 'pop_art',
                'قدیمی': 'old_photo',
                'تلویزیون قدیمی': 'vhs',
                'پولاروید': 'polaroid',
                'روشن کردن': 'sharpen',
                'نویز': 'noise',
                'دانه فیلم': 'film_grain',
                'قطره آب': 'rain',
                'پیکسلی': 'pixelate',
                'وینیت': 'vignette'
            };
            
            filterName = filterMap[filterName] || filterName.toLowerCase().replace(/\s+/g, '_');
            
            console.log('Selecting filter:', filterName);
            
            if (socket && socket.connected) {
                socket.emit('set_filter', { filter: filterName });
                
                // آپدیت UI
                filterButtons.forEach(b => b.classList.remove('active'));
                this.classList.add('active');
            }
        });
    });
    
    // دکمه اول (اصلی) را به صورت پیش‌فرض فعال کنیم
    if (filterButtons.length > 0) {
        filterButtons[0].classList.add('active');
    }
});
