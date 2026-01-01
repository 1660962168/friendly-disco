// --- 1. 初始化 Lucide 图标 ---
lucide.createIcons();

// --- 2. 状态管理 ---
const state = {
    total: 1247,
    success: 1198,
    failed: 49,
    avgTime: 0.85,
    logs: [
        { id: 1, time: '14:32:15', plate: '京A88888', status: 'success', confidence: 98.5, location: '入口A' },
        { id: 2, time: '14:31:42', plate: '沪B66666', status: 'success', confidence: 96.2, location: '入口B' },
        { id: 3, time: '14:30:18', plate: '粤C12345', status: 'failed', confidence: 62.3, location: '入口A' },
        { id: 4, time: '14:29:55', plate: '浙D77777', status: 'success', confidence: 99.1, location: '入口C' },
        { id: 5, time: '14:28:33', plate: '苏E99999', status: 'success', confidence: 97.8, location: '入口A' },
    ]
};

// --- 3. DOM 元素 ---
const els = {
    scanLine: document.getElementById('scan-line'),
    recBox: document.getElementById('recognition-box'),
    plateResult: document.getElementById('plate-result'),
    statusBadge: document.getElementById('status-badge'),
    charContainer: document.getElementById('char-container'),
    logList: document.getElementById('log-list'),
    // Stats
    statTotal: document.getElementById('stat-total'),
    statSuccess: document.getElementById('stat-success'),
    statFailed: document.getElementById('stat-failed'),
    statTime: document.getElementById('stat-time'),
    statRate: document.getElementById('stat-rate'),
    // Char details
    charCount: document.getElementById('char-count'),
    charConf: document.getElementById('char-conf'),
    processTime: document.getElementById('process-time'),
    // Logs
    logTotal: document.getElementById('log-total-count'),
    logSuccess: document.getElementById('log-success-count'),
    logFail: document.getElementById('log-fail-count'),
    otherView: document.getElementById('other-view'),
    videoModeTitle: document.getElementById('video-mode-title'),
    btnImage: document.getElementById('btn-image'),
    btnVideo: document.getElementById('btn-video'),
};


function randomPlate() {
    const provinces = ['京', '津', '沪', '渝', '冀', '豫', '云', '辽', '黑', '湘', '皖', '鲁', '新', '苏', '浙', '赣', '鄂', '桂', '甘', '晋', '蒙', '陕', '吉', '闽', '贵', '粤', '青', '藏', '川', '宁', '琼'];
    const letters = 'ABCDEFGHJKLMNPQRSTUVWXYZ';
    const numbers = '0123456789';

    const p = provinces[Math.floor(Math.random() * provinces.length)];
    const l = letters[Math.floor(Math.random() * letters.length)];
    let rest = '';
    for (let i = 0; i < 5; i++) {
        rest += Math.random() > 0.5
            ? letters[Math.floor(Math.random() * letters.length)]
            : numbers[Math.floor(Math.random() * numbers.length)];
    }
    return p + l + rest;
}

// --- 5. 渲染逻辑 ---

// 渲染日志
function renderLogs() {
    els.logList.innerHTML = state.logs.map((log, index) => `
    <div class="rounded-xl bg-gradient-to-r 
        ${log.status === 'success'
            ? 'from-emerald-500/10 to-transparent border-emerald-500/20 dark:from-emerald-500/5' // 成功状态
            : 'from-red-500/10 to-transparent border-red-500/20 dark:from-red-500/5'             // 失败状态
        } 
        border p-4 hover:bg-gray-50 dark:hover:bg-white/5 transition-colors ${index === 0 ? 'animate-[slideIn_0.3s_ease-out]' : ''}">
        
        <div class="flex items-start justify-between mb-2">
            <div class="flex items-center gap-3">
                <div class="w-2 h-2 rounded-full ${log.status === 'success' ? 'bg-emerald-500' : 'bg-red-500'}"></div>
                <span class="text-sm text-gray-500 dark:text-gray-400">${log.time}</span>
            </div>
            </div>
        
        <div class="flex items-center justify-between">
            <div>
                <div class="text-lg tracking-wider mb-1  text-gray-800 dark:text-gray-200">${log.plate}</div>
                </div>
        </div>
    </div>
`).join('');

    // 更新日志统计
    els.logTotal.innerText = state.logs.length;
    els.logSuccess.innerText = state.logs.filter(l => l.status === 'success').length;
    els.logFail.innerText = state.logs.filter(l => l.status === 'failed').length;
}

// 渲染字符分割
function renderSegmentation(plate) {
    if (!plate) {
        els.charContainer.innerHTML = `
                    <div class="flex items-center justify-center text-gray-500">
                        <div class="text-center">
                            <i data-lucide="grid-3x3" class="w-12 h-12 mx-auto mb-3 opacity-30"></i>
                            <p>等待车牌识别...</p>
                        </div>
                    </div>`;
        lucide.createIcons();
        return;
    }

    const confidences = Array.from({ length: plate.length }, () => 85 + Math.random() * 14);
    const avgConf = (confidences.reduce((a, b) => a + b, 0) / confidences.length).toFixed(1);

    els.charContainer.innerHTML = plate.split('').map((char, index) => `
    <div class="relative group flex-1 max-w-[110px] min-w-[50px] aspect-[3/4] animate-[slideIn_0.3s_ease-out_backwards]" style="animation-delay: ${index * 0.05}s,margin-bottom: 10px;">
        <div class="w-full h-full rounded-xl bg-gradient-to-br from-gray-800 to-gray-900 border-2 border-cyan-500/50 flex items-center justify-center text-3xl sm:text-5xl lg:text-6xl font-bold shadow-lg shadow-cyan-500/20 transition-all duration-300 group-hover:scale-105 group-hover:-translate-y-2 group-hover:border-cyan-400 group-hover:shadow-cyan-400/40 cursor-default">
            ${char}
        </div>
        
        <div class="absolute -bottom-8 left-1/2 -translate-x-1/2 w-full text-center opacity-70 group-hover:opacity-100 transition-opacity">
            <span class="inline-block px-2 py-0.5 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-xs sm:text-sm text-emerald-400 font-mono">
                ${confidences[index].toFixed(1)}%
            </span>
        </div>
    </div>
`).join('');

    els.charCount.innerText = plate.length;
    els.charConf.innerText = avgConf + '%';
    els.processTime.innerText = (0.5 + Math.random() * 0.5).toFixed(2);
}

// --- 6. 模拟主循环 ---
function simulationLoop() {
    // 随机触发识别
    if (Math.random() > 0.6) {
        // 1. 开始处理状态
        els.statusBadge.innerText = "识别中...";
        els.statusBadge.className = "px-3 py-1.5 rounded-lg text-xs bg-cyan-50 text-cyan-600 border border-cyan-200 dark:bg-cyan-500/20 dark:text-cyan-400 dark:border-cyan-500/30";
        els.scanLine.classList.remove('hidden');
        els.recBox.classList.add('hidden');

        // 2. 延迟显示结果
        setTimeout(() => {
            const newPlate = randomPlate();

            // 更新视频流上的结果
            els.statusBadge.innerText = "识别成功";
            els.statusBadge.className = "px-3 py-1.5 rounded-lg text-xs bg-emerald-50 text-emerald-600 border border-emerald-200 dark:bg-emerald-500/20 dark:text-emerald-400 dark:border-emerald-500/30";
            els.scanLine.classList.add('hidden');
            els.recBox.classList.remove('hidden');
            els.plateResult.innerText = newPlate;

            // 更新统计
            state.total++;
            const isSuccess = Math.random() > 0.1;
            if (isSuccess) state.success++; else state.failed++;
            state.avgTime = (0.7 + Math.random() * 0.3).toFixed(2);

            els.statTotal.innerText = state.total.toLocaleString();
            els.statSuccess.innerText = state.success.toLocaleString();
            els.statFailed.innerText = state.failed;
            els.statTime.innerText = state.avgTime + 's';
            els.statRate.innerText = ((state.success / state.total) * 100).toFixed(1) + '%';

            // 添加日志
            const newLog = {
                id: Date.now(),
                time: new Date().toLocaleTimeString('zh-CN', { hour12: false }),
                plate: newPlate,
                status: isSuccess ? 'success' : 'failed',
                confidence: 85 + Math.random() * 14,
                location: ['入口A', '入口B', '入口C'][Math.floor(Math.random() * 3)]
            };
            state.logs.unshift(newLog);
            if (state.logs.length > 20) state.logs.pop();
            renderLogs();

            // 更新字符可视化
            renderSegmentation(newPlate);

            // 1.5秒后回到待机
            setTimeout(() => {
                els.statusBadge.innerText = "待机中";
                els.statusBadge.className = "px-3 py-1.5 rounded-lg text-xs bg-gray-100 text-gray-500 border border-gray-200 dark:bg-gray-800/50 dark:text-gray-400 dark:border-gray-700/30";
                els.recBox.classList.add('hidden');
            }, 1500);

        }, 1500);
    }
}

// 启动模拟
renderLogs();
// setInterval(simulationLoop, 4000);

// --- 7. 图表初始化 (Chart.js) ---

// 1. 趋势图 (Trend Chart)
const ctxTrend = document.getElementById('trendChart').getContext('2d');
const trendGradient = ctxTrend.createLinearGradient(0, 0, 0, 300);
trendGradient.addColorStop(0, 'rgba(34, 211, 238, 0.3)');
trendGradient.addColorStop(1, 'rgba(34, 211, 238, 0)');

new Chart(ctxTrend, {
    type: 'line',
    data: {
        labels: Array.from({ length: 24 }, (_, i) => `${i.toString().padStart(2, '0')}:00`),
        datasets: [{
            label: '识别数量',
            data: Array.from({ length: 24 }, () => Math.floor(Math.random() * 100) + 20),
            borderColor: '#22d3ee',
            backgroundColor: trendGradient,
            borderWidth: 2,
            fill: true,
            tension: 0.4,
            pointRadius: 0
        }]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: { display: false },
            tooltip: {
                // Tooltip 也可以改得稍微透明一点，适配两种模式
                backgroundColor: 'rgba(31, 41, 55, 0.9)',
                titleColor: '#fff',
                bodyColor: '#fff',
                borderColor: 'rgba(55, 65, 81, 0.5)',
                borderWidth: 1
            }
        },
        scales: {
            x: {
                // 【修改点】将 grid 颜色改为低透明度的灰色
                grid: { color: 'rgba(128, 128, 128, 0.1)' },
                ticks: { color: '#6b7280', maxTicksLimit: 8 }
            },
            y: {
                // 【修改点】同上
                grid: { color: 'rgba(128, 128, 128, 0.1)' },
                ticks: { color: '#6b7280' }
            }
        }
    }
});

// 2. 柱状图 (Bar Chart)
const ctxBar = document.getElementById('barChart').getContext('2d');
new Chart(ctxBar, {
    type: 'bar',
    data: {
        labels: ['蓝牌', '黄牌', '绿牌'],
        datasets: [{
            label: '数量',
            data: [856, 234, 123],
            backgroundColor: ['#3b82f6', '#eab308', '#10b981'],
            borderRadius: 4,
            barThickness: 40
        }]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: { display: false },
            tooltip: {
                backgroundColor: 'rgba(31, 41, 55, 0.9)',
                borderColor: 'rgba(55, 65, 81, 0.5)',
                borderWidth: 1
            }
        },
        scales: {
            x: {
                grid: { display: false },
                ticks: { color: '#6b7280' }
            },
            y: {
                // 【修改点】将 grid 颜色改为低透明度的灰色
                grid: { color: 'rgba(128, 128, 128, 0.1)' },
                ticks: { color: '#6b7280' }
            }
        }
    }
});
// --- 8. 交互函数 (Global scope) ---
window.toggleSettings = function (show) {
    const modal = document.getElementById('settings-modal');
    if (show) {
        modal.classList.remove('hidden');
    } else {
        modal.classList.add('hidden');
    }
}

// --- [新增] 文件选择处理逻辑 ---
window.handleFileSelect = function(input, type) {
    if (input.files && input.files[0]) {
        const file = input.files[0];
        
        if (type === 'image') {
            const reader = new FileReader();
            reader.onload = function(e) {
                const img = document.getElementById('image-preview');
                const empty = document.getElementById('image-empty');
                const overlay = document.getElementById('image-overlay');
                
                img.src = e.target.result;
                img.classList.remove('hidden');
                empty.classList.add('hidden');
                overlay.classList.remove('hidden');
            }
            reader.readAsDataURL(file);
            
            // 这里可以添加调用后端识别 API 的代码
            // console.log("Image selected, ready to upload...");
            
        } else if (type === 'video') {
            const videoUrl = URL.createObjectURL(file);
            const video = document.getElementById('video-player');
            const empty = document.getElementById('video-empty');
            const closeBtn = document.getElementById('video-close-btn');
            const inputEl = document.getElementById('video-upload');

            video.src = videoUrl;
            video.classList.remove('hidden');
            empty.classList.add('hidden');
            closeBtn.classList.remove('hidden');
            
            // 隐藏 input 以便用户可以点击视频控件（播放/暂停）
            inputEl.classList.add('hidden'); 
            
            // 自动播放
            // video.play();
        }
    }
}

// --- [新增] 重置视频逻辑 ---
window.resetVideo = function() {
    const video = document.getElementById('video-player');
    const empty = document.getElementById('video-empty');
    const closeBtn = document.getElementById('video-close-btn');
    const inputEl = document.getElementById('video-upload');

    video.pause();
    video.src = ""; // 释放资源
    video.classList.add('hidden');
    empty.classList.remove('hidden');
    closeBtn.classList.add('hidden');
    
    // 恢复 input 覆盖，允许再次上传
    inputEl.value = "";
    inputEl.classList.remove('hidden');
}

// --- [修改] 切换模式逻辑 (完善版) ---
window.switchVideoMode = function (mode) {
    const titleContainer = document.getElementById('mode-title-container');
    const btnImage = document.getElementById('btn-image');
    const btnVideo = document.getElementById('btn-video');
    
    // 1. 更新标题和图标 (直接重写 innerHTML 以确保 Lucide 图标正确更新)
    if (mode === 'image') {
        titleContainer.innerHTML = `
            <i data-lucide="image" class="w-5 h-5 text-cyan-400"></i>
            <h2 class="text-lg font-medium">图片识别</h2>
        `;
    } else if (mode === 'video') {
        titleContainer.innerHTML = `
            <i data-lucide="video" class="w-5 h-5 text-cyan-400"></i>
            <h2 class="text-lg font-medium">视频识别</h2>
        `;
    }
    // 重新渲染新插入的图标
    lucide.createIcons();

    // 2. 更新按钮样式
    // 辅助函数：未选中样式
    const setInactive = (btn) => {
        btn.className = "px-4 py-2 rounded-md text-sm transition-all text-gray-500 hover:text-gray-900 dark:text-gray-400 dark:hover:text-gray-300 border border-transparent";
    };
    // 辅助函数：选中样式
    const setActive = (btn) => {
        btn.className = "px-4 py-2 rounded-md text-sm transition-all bg-white text-cyan-600 border border-cyan-200 shadow-sm dark:bg-cyan-500/20 dark:text-cyan-400 dark:border-cyan-500/50 dark:shadow-[0_0_10px_rgba(34,211,238,0.2)]";
    };

    if (mode === 'image') {
        setActive(btnImage);
        setInactive(btnVideo);
    } else {
        setInactive(btnImage);
        setActive(btnVideo);
    }

    // 3. 切换显示区域
    const imgView = document.getElementById('image-view');
    const vidView = document.getElementById('video-view');
    
    if (mode === 'image') {
        imgView.classList.remove('hidden');
        imgView.classList.add('flex');
        vidView.classList.add('hidden');
        vidView.classList.remove('block');
        
        // 切换回图片模式时，如果有正在播放的视频，建议暂停
        const video = document.getElementById('video-player');
        if(video) video.pause();
        
    } else if (mode === 'video') {
        imgView.classList.add('hidden');
        imgView.classList.remove('flex');
        vidView.classList.remove('hidden');
        vidView.classList.add('block');
    }
}
function updatePerformance() {
    // 请求 Flask 后端接口
    fetch('/api/performance')
        .then(response => response.json())
        .then(data => {
            // 更新 CPU
            const cpuVal = data.cpu_usage;
            document.getElementById('cpu-text').innerText = cpuVal + '%';
            document.getElementById('cpu-bar').style.width = cpuVal + '%';

            // 更新 内存
            const memVal = data.memory_usage;
            document.getElementById('memory-text').innerText = memVal + '%';
            document.getElementById('memory-bar').style.width = memVal + '%';

            // 更新 显卡
            const gpuVal = data.gpu_usage;
            document.getElementById('gpu-text').innerText = gpuVal + '%';
            document.getElementById('gpu-bar').style.width = gpuVal + '%';
        })
        .catch(error => {
            console.error('获取性能数据失败:', error);
        });
}

// 1. 页面加载完成后立即调用一次
// updatePerformance();

// 2. 之后每隔 2000 毫秒 (2秒) 自动刷新一次
// setInterval(updatePerformance, 2000);

// --- 9. 主题切换逻辑 ---
function initTheme() {
    // 检查本地存储或系统偏好
    if (localStorage.theme === 'dark' || (!('theme' in localStorage) && window.matchMedia('(prefers-color-scheme: dark)').matches)) {
        document.documentElement.classList.add('dark');
    } else {
        document.documentElement.classList.remove('dark');
    }
    // 刷新图标（如果需要重新渲染）
    if (typeof lucide !== 'undefined') lucide.createIcons();
}

window.toggleTheme = function () {
    const html = document.documentElement;
    if (html.classList.contains('dark')) {
        html.classList.remove('dark');
        localStorage.theme = 'light';
    } else {
        html.classList.add('dark');
        localStorage.theme = 'dark';
    }
    // 强制重新渲染图标以确保 sun/moon 切换显示正确（虽然 CSS hidden 也可以处理）
    lucide.createIcons();
}

// 初始化调用
initTheme();

// --- 图片上传与识别逻辑 ---

window.handleImageUpload = function(input) {
    if (input.files && input.files[0]) {
        const file = input.files[0];
        
        const uploadArea = document.getElementById('image-upload-area');
        const resultContainer = document.getElementById('image-result-container');
        const loading = document.getElementById('image-loading');
        const previewImg = document.getElementById('image-preview');
        
        // 状态显示的元素
        const statusBadge = document.getElementById('status-badge');
        const charConfEl = document.getElementById('char-conf');
        // 可选：你可以在HTML里加一个专门显示车牌号的文字区域，比如 id="plate-number-display"
        
        uploadArea.classList.add('hidden'); 
        resultContainer.classList.remove('hidden');
        loading.classList.remove('hidden'); // 显示 "AI 正在识别中..."

        const formData = new FormData();
        formData.append('file', file);

        // --- 第一步：请求 YOLO 检测 ---
        fetch('/api/detect/yolo', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // 1. YOLO 成功：立刻显示带框的图片
                previewImg.src = data.result_url + '?t=' + new Date().getTime();
                loading.classList.add('hidden'); // 隐藏全屏 Loading，让用户看到图

                if (data.has_plate && data.crop_filename) {
                    // 更新状态：正在识别文字
                    if(statusBadge) {
                        statusBadge.innerText = "定位成功，正在分析文字...";
                        statusBadge.className = "px-3 py-1.5 rounded-lg text-xs bg-blue-50 text-blue-600 border border-blue-200 animate-pulse";
                    }
                    if(charConfEl) charConfEl.innerText = "--%";

                    // --- 第二步：请求 OCR 识别 ---
                    // 这里传入 YOLO 返回的文件名
                    startOcrRequest(data.crop_filename); 

                } else {
                    // 没检测到车牌
                    if(statusBadge) {
                        statusBadge.innerText = "未检测到车牌";
                        statusBadge.className = "px-3 py-1.5 rounded-lg text-xs bg-yellow-50 text-yellow-600 border border-yellow-200";
                    }
                }
            } else {
                alert('YOLO 识别出错: ' + data.error);
                resetImageUpload();
            }
        })
        .catch(err => {
            console.error(err);
            alert('网络请求失败');
            resetImageUpload();
            loading.classList.add('hidden');
        });
    }
}

function startOcrRequest(cropFilename) {
    fetch('/api/detect/ocr', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ crop_filename: cropFilename })
    })
    .then(response => response.json())
    .then(data => {
        const statusBadge = document.getElementById('status-badge');
        const charConfEl = document.getElementById('char-conf');
        
        if (data.success) {
            console.log("OCR 结果:", data.plate_text);
            
            // 更新文字显示
            if(statusBadge) {
                statusBadge.innerText = data.plate_text ? `识别成功: ${data.plate_text}` : "文字识别失败";
                statusBadge.className = data.plate_text 
                    ? "px-3 py-1.5 rounded-lg text-xs bg-emerald-50 text-emerald-600 border border-emerald-200"
                    : "px-3 py-1.5 rounded-lg text-xs bg-orange-50 text-orange-600 border border-orange-200";
            }

            // 更新置信度
            if (charConfEl && data.confidence) {
                charConfEl.innerText = (data.confidence * 100).toFixed(1) + '%';
                charConfEl.className = data.confidence > 0.8 
                            ? "text-2xl text-emerald-400 font-mono" 
                            : "text-2xl text-yellow-400 font-mono";
            }
        }
    })
    .catch(err => {
        console.error("OCR 请求失败:", err);
    });
}

// 重置界面，允许再次上传
window.resetImageUpload = function() {
    document.getElementById('image-upload').value = ""; // 清空 input
    document.getElementById('image-upload-area').classList.remove('hidden');
    document.getElementById('image-result-container').classList.add('hidden');
    document.getElementById('status-badge').innerText = "待机中";
}

// --- 完善 switchVideoMode 函数 ---
window.switchVideoMode = function (mode) {
    const titleContainer = document.getElementById('mode-title-container');
    const btnImage = document.getElementById('btn-image');
    const btnVideo = document.getElementById('btn-video');
    const imgView = document.getElementById('image-view');
    const vidView = document.getElementById('video-view');

    // 1. 更新标题和图标
    if (titleContainer) {
        if (mode === 'image') {
            titleContainer.innerHTML = `<i data-lucide="image" class="w-5 h-5 text-cyan-400"></i><h2 class="text-lg font-medium">图片识别</h2>`;
        } else {
            titleContainer.innerHTML = `<i data-lucide="video" class="w-5 h-5 text-cyan-400"></i><h2 class="text-lg font-medium">视频识别</h2>`;
        }
        lucide.createIcons();
    }

    // 2. 更新按钮状态
    const setBtnStyle = (btn, active) => {
        if (!btn) return;
        if (active) {
            btn.className = "px-4 py-2 rounded-md text-sm transition-all bg-white text-cyan-600 border border-cyan-200 shadow-sm dark:bg-cyan-500/20 dark:text-cyan-400 dark:border-cyan-500/50 dark:shadow-[0_0_10px_rgba(34,211,238,0.2)]";
        } else {
            btn.className = "px-4 py-2 rounded-md text-sm transition-all text-gray-500 hover:text-gray-900 dark:text-gray-400 dark:hover:text-gray-300 border border-transparent";
        }
    };

    setBtnStyle(btnImage, mode === 'image');
    setBtnStyle(btnVideo, mode === 'video');

    // 3. 切换显示区域
    if (imgView && vidView) {
        if (mode === 'image') {
            imgView.parentElement.classList.remove('hidden'); // 确保父级可见
            imgView.style.display = 'flex';
            vidView.style.display = 'none';
        } else {
            imgView.style.display = 'none';
            vidView.style.display = 'flex';
            vidView.classList.remove('hidden');
        }
    }
}