// --- 1. 初始化 Lucide 图标 ---
lucide.createIcons();

// --- 2. 状态管理 ---
const state = {
    logs: [] // 数据将从后端获取
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
    // Buttons
    btnImage: document.getElementById('btn-image'),
    btnVideo: document.getElementById('btn-video'),
    // [新增] 搜索和下载元素
    searchInput: document.getElementById('history-search-input'),
    btnDownload: document.getElementById('btn-download-history'),
};

// --- 5. 渲染逻辑 ---

// 渲染日志
function renderLogs() {
    if (!state.logs || state.logs.length === 0) {
        els.logList.innerHTML = '<div class="text-center text-gray-400 py-4">暂无记录</div>';
        return;
    }

    els.logList.innerHTML = state.logs.map((log, index) => `
    <div class="rounded-xl bg-gradient-to-r 
        ${log.status === 'success'
            ? 'from-emerald-500/10 to-transparent border-emerald-500/20 dark:from-emerald-500/5' 
            : 'from-red-500/10 to-transparent border-red-500/20 dark:from-red-500/5'             
        } 
        border p-4 hover:bg-gray-50 dark:hover:bg-white/5 transition-colors ${index === 0 ? 'animate-[slideIn_0.3s_ease-out]' : ''}">
        
        <div class="flex items-start justify-between mb-2">
            <div class="flex items-center gap-3">
                <div class="w-2 h-2 rounded-full ${log.status === 'success' ? 'bg-emerald-500' : 'bg-red-500'}"></div>
                <span class="text-sm text-gray-500 dark:text-gray-400">${log.time}</span>
            </div>
            <span class="text-xs text-gray-400">用时: ${log.duration}s</span>
        </div>
        
        <div class="flex items-center justify-between">
            <div>
                <div class="text-lg tracking-wider mb-1 text-gray-800 dark:text-gray-200">${log.plate}</div>
            </div>
            <span class="text-xs ${log.confidence > 80 ? 'text-emerald-500' : 'text-yellow-500'}">成功率: ${log.confidence}%</span>
        </div>
    </div>
`).join('');
}

// 渲染字符分割 (修复：适配白天/黑夜模式)
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
        if(els.charCount) els.charCount.innerText = "0";
        return;
    }

    // 模拟每个字符的置信度
    const confidences = Array.from({ length: plate.length }, () => 90 + Math.random() * 9);
    
    els.charContainer.innerHTML = plate.split('').map((char, index) => `
    <div class="relative group flex-1 max-w-[110px] min-w-[50px] aspect-[3/4] animate-[slideIn_0.3s_ease-out_backwards]" style="animation-delay: ${index * 0.05}s; margin-bottom: 10px;">
        <div class="w-full h-full rounded-xl border-2 flex items-center justify-center text-3xl sm:text-5xl lg:text-6xl font-bold transition-all duration-300 cursor-default
            bg-white border-gray-200 text-gray-800 shadow-lg shadow-gray-200/50
            dark:bg-gradient-to-br dark:from-gray-800 dark:to-gray-900 dark:border-cyan-500/50 dark:text-gray-100 dark:shadow-cyan-500/20
            
            group-hover:scale-105 group-hover:-translate-y-2 
            group-hover:border-cyan-500 dark:group-hover:border-cyan-400
            group-hover:shadow-xl dark:group-hover:shadow-cyan-400/40">
            ${char}
        </div>
        
        <div class="absolute -bottom-8 left-1/2 -translate-x-1/2 w-full text-center opacity-70 group-hover:opacity-100 transition-opacity">
            <span class="inline-block px-2 py-0.5 rounded-full text-xs sm:text-sm font-mono border
                bg-emerald-50 text-emerald-600 border-emerald-200
                dark:bg-emerald-500/10 dark:text-emerald-400 dark:border-emerald-500/20">
                ${confidences[index].toFixed(1)}%
            </span>
        </div>
    </div>
`).join('');

    if(els.charCount) els.charCount.innerText = plate.length;
}


// --- 7. 图表初始化 (Chart.js) ---

// 1. 趋势图
const ctxTrend = document.getElementById('trendChart').getContext('2d');
const trendGradient = ctxTrend.createLinearGradient(0, 0, 0, 300);
trendGradient.addColorStop(0, 'rgba(34, 211, 238, 0.3)');
trendGradient.addColorStop(1, 'rgba(34, 211, 238, 0)');

// 保存实例以便更新
window.trendChart = new Chart(ctxTrend, {
    type: 'line',
    data: {
        labels: [], // 等待 API 数据
        datasets: [{
            label: '识别数量',
            data: [],
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
        plugins: { legend: { display: false } },
        scales: {
            x: { grid: { color: 'rgba(128, 128, 128, 0.1)' }, ticks: { color: '#6b7280', maxTicksLimit: 8 } },
            y: { grid: { color: 'rgba(128, 128, 128, 0.1)' }, ticks: { color: '#6b7280' } }
        }
    }
});

// 2. 柱状图
const ctxBar = document.getElementById('barChart').getContext('2d');
// 保存实例
window.barChart = new Chart(ctxBar, {
    type: 'bar',
    data: {
        labels: [],
        datasets: [{
            label: '数量',
            data: [],
            backgroundColor: [],
            borderRadius: 4,
            barThickness: 40
        }]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
            x: { grid: { display: false }, ticks: { color: '#6b7280' } },
            y: { grid: { color: 'rgba(128, 128, 128, 0.1)' }, ticks: { color: '#6b7280' } }
        }
    }
});

// --- 新增：数据获取与更新函数 ---

function updateTrend() {
    fetch('/api/stats/trend')
        .then(res => res.json())
        .then(data => {
            window.trendChart.data.labels = data.labels;
            window.trendChart.data.datasets[0].data = data.data;
            window.trendChart.update();
        })
        .catch(console.error);
}

function updateDistribution() {
    fetch('/api/stats/distribution')
        .then(res => res.json())
        .then(data => {
            window.barChart.data.labels = data.labels;
            window.barChart.data.datasets[0].data = data.data;
            window.barChart.data.datasets[0].backgroundColor = data.colors;
            window.barChart.update();
        })
        .catch(console.error);
}

// 支持搜索参数
function updateHistoryAndStats(searchQuery = "") {
    // 构建 URL，如果有搜索关键词则附加参数
    let url = '/api/stats/history';
    if (searchQuery) {
        url += `?search=${encodeURIComponent(searchQuery)}`;
    }

    fetch(url)
        .then(res => res.json())
        .then(data => {
            // 更新列表
            state.logs = data.logs;
            renderLogs();
            
            // 更新统计数字
            if(data.stats) {
                if(els.statTotal) els.statTotal.innerText = data.stats.total;
                if(els.statSuccess) els.statSuccess.innerText = data.stats.success;
                if(els.statFailed) els.statFailed.innerText = data.stats.failed;
                
                // 更新平均用时
                if(els.statTime) els.statTime.innerText = data.stats.avg_time + 's';

                // 计算成功率
                const rate = data.stats.total > 0 
                    ? ((data.stats.success / data.stats.total) * 100).toFixed(1) 
                    : "0.0";
                if(els.statRate) els.statRate.innerText = rate + "%";
                
                // 底部 Log 统计
                if(els.logTotal) els.logTotal.innerText = data.stats.total;
                if(els.logSuccess) els.logSuccess.innerText = data.stats.success;
                if(els.logFail) els.logFail.innerText = data.stats.failed;
            }
        })
        .catch(console.error);
}

// --- 性能监控 ---
function updatePerformance() {
    fetch('/api/performance')
        .then(response => response.json())
        .then(data => {
            const cpuVal = data.cpu_usage;
            document.getElementById('cpu-text').innerText = cpuVal + '%';
            document.getElementById('cpu-bar').style.width = cpuVal + '%';

            const memVal = data.memory_usage;
            document.getElementById('memory-text').innerText = memVal + '%';
            document.getElementById('memory-bar').style.width = memVal + '%';

            const gpuVal = data.gpu_usage;
            document.getElementById('gpu-text').innerText = gpuVal + '%';
            document.getElementById('gpu-bar').style.width = gpuVal + '%';
        })
        .catch(error => {
            console.error('获取性能数据失败:', error);
        });
}

// --- 图片上传与识别逻辑 ---

window.handleImageUpload = function(input) {
    if (input.files && input.files[0]) {
        const file = input.files[0];
        
        const uploadArea = document.getElementById('image-upload-area');
        const resultContainer = document.getElementById('image-result-container');
        const loading = document.getElementById('image-loading');
        const previewImg = document.getElementById('image-preview');
        const statusBadge = document.getElementById('status-badge');
        const charConfEl = document.getElementById('char-conf');
        
        uploadArea.classList.add('hidden'); 
        resultContainer.classList.remove('hidden');
        loading.classList.remove('hidden');

        const formData = new FormData();
        formData.append('file', file);

        // 1. YOLO 检测
        fetch('/api/detect/yolo', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                previewImg.src = data.result_url + '?t=' + new Date().getTime();
                loading.classList.add('hidden');

                if (data.has_plate && data.crop_filename) {
                    if(statusBadge) {
                        statusBadge.innerText = "定位成功，正在分析文字...";
                        statusBadge.className = "px-3 py-1.5 rounded-lg text-xs bg-blue-50 text-blue-600 border border-blue-200 animate-pulse";
                    }
                    if(charConfEl) charConfEl.innerText = "--%";

                    // 2. OCR 识别
                    startOcrRequest(data.crop_filename); 

                } else {
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
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ crop_filename: cropFilename })
    })
    .then(response => response.json())
    .then(data => {
        const statusBadge = document.getElementById('status-badge');
        const charConfEl = document.getElementById('char-conf');
        const processTimeEl = document.getElementById('process-time');
        
        if (data.success) {
            console.log("OCR 结果:", data.plate_text);
            
            // 更新 UI 状态
            if(statusBadge) {
                statusBadge.innerText = data.plate_text ? `识别成功: ${data.plate_text}` : "文字识别失败";
                statusBadge.className = data.plate_text 
                    ? "px-3 py-1.5 rounded-lg text-xs bg-emerald-50 text-emerald-600 border border-emerald-200"
                    : "px-3 py-1.5 rounded-lg text-xs bg-orange-50 text-orange-600 border border-orange-200";
            }

            if (charConfEl && data.confidence) {
                charConfEl.innerText = (data.confidence * 100).toFixed(1) + '%';
                charConfEl.className = data.confidence > 0.8 
                            ? "text-2xl text-emerald-400 font-mono" 
                            : "text-2xl text-yellow-400 font-mono";
            }
            
            // 显示识别用时
            if (processTimeEl && data.duration) {
                processTimeEl.innerText = data.duration;
            }

            // [核心] 渲染分割字符
            renderSegmentation(data.plate_text);

            // [核心] 刷新所有统计数据
            updateTrend();
            updateDistribution();
            updateHistoryAndStats();

        } else {
            // 失败处理
            if(statusBadge) statusBadge.innerText = "识别失败";
            renderSegmentation(null);
        }
    })
    .catch(err => {
        console.error("OCR 请求失败:", err);
    });
}

// 重置与模式切换逻辑
window.resetImageUpload = function() {
    document.getElementById('image-upload').value = ""; 
    document.getElementById('image-upload-area').classList.remove('hidden');
    document.getElementById('image-result-container').classList.add('hidden');
    document.getElementById('status-badge').innerText = "待机中";
    renderSegmentation(null);
}

// 其他辅助函数
window.toggleSettings = function (show) {
    const modal = document.getElementById('settings-modal');
    show ? modal.classList.remove('hidden') : modal.classList.add('hidden');
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
    lucide.createIcons();
}

function initTheme() {
    if (localStorage.theme === 'dark' || (!('theme' in localStorage) && window.matchMedia('(prefers-color-scheme: dark)').matches)) {
        document.documentElement.classList.add('dark');
    }
    lucide.createIcons();
}
initTheme();

window.switchVideoMode = function (mode) {
    const imgView = document.getElementById('image-view');
    const vidView = document.getElementById('video-view');
    const titleContainer = document.getElementById('mode-title-container');
    const btnImage = document.getElementById('btn-image');
    const btnVideo = document.getElementById('btn-video');

    const setBtnActive = (btn, active) => {
        btn.className = active 
            ? "px-4 py-2 rounded-md text-sm transition-all bg-white text-cyan-600 border border-cyan-200 shadow-sm dark:bg-cyan-500/20 dark:text-cyan-400 dark:border-cyan-500/50 dark:shadow-[0_0_10px_rgba(34,211,238,0.2)]"
            : "px-4 py-2 rounded-md text-sm transition-all text-gray-500 hover:text-gray-900 dark:text-gray-400 dark:hover:text-gray-300 border border-transparent";
    };

    if (mode === 'image') {
        titleContainer.innerHTML = `<i data-lucide="image" class="w-5 h-5 text-cyan-400"></i><h2 class="text-lg font-medium">图片识别</h2>`;
        imgView.style.display = 'flex';
        vidView.style.display = 'none';
        setBtnActive(btnImage, true);
        setBtnActive(btnVideo, false);
    } else {
        titleContainer.innerHTML = `<i data-lucide="video" class="w-5 h-5 text-cyan-400"></i><h2 class="text-lg font-medium">视频识别</h2>`;
        imgView.style.display = 'none';
        vidView.style.display = 'flex';
        vidView.classList.remove('hidden');
        setBtnActive(btnImage, false);
        setBtnActive(btnVideo, true);
    }
    lucide.createIcons();
}

// --- 事件绑定 ---

// 1. 实时搜索功能
if (els.searchInput) {
    els.searchInput.addEventListener('input', function(e) {
        const query = e.target.value.trim();
        // 调用更新函数，传入搜索词
        updateHistoryAndStats(query);
    });
}

// 2. 历史记录下载功能
if (els.btnDownload) {
    els.btnDownload.addEventListener('click', function() {
        window.location.href = '/api/download/history';
    });
}

// --- 初始化调用 ---
updateTrend();
updateDistribution();
updateHistoryAndStats();
updatePerformance();

// 启用定时刷新
setInterval(updatePerformance, 2000);
fetch('/api/init/ocr')
    .then(res => res.json())
    .then(data => {
        if(data.success) {
            console.log("OCR 模型后台预加载完成");
        }
    })
    .catch(console.error);



const confInput = document.getElementById('conf-threshold-input');
const confDisplay = document.getElementById('conf-threshold-display');

if (confInput && confDisplay) {
    confInput.addEventListener('input', function(e) {
        // 实时更新右侧数字
        confDisplay.innerText = e.target.value + '%';
    });
}

// 2. 保存设置函数
window.saveSettings = function() {
    const val = document.getElementById('conf-threshold-input').value;
    
    fetch('/api/settings/update', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ confidence_threshold: val })
    })
    .then(res => res.json())
    .then(data => {
        if (data.success) {
            // 保存成功后关闭弹窗
            toggleSettings(false);
        } else {
            alert("保存失败：" + data.error);
        }
    })
    .catch(err => {
        console.error(err);
        alert("网络请求失败");
    });
}