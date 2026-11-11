"use strict";

const $ = (selector) => document.querySelector(selector);

const elements = {
  modelPill: $("#modelPill"),
  modelStatus: $("#modelStatus"),
  dropzone: $("#dropzone"),
  dropIdle: $("#dropIdle"),
  previewStage: $("#previewStage"),
  previewImage: $("#previewImage"),
  previewName: $("#previewName"),
  fileInput: $("#fileInput"),
  cameraInput: $("#cameraInput"),
  pickButton: $("#pickButton"),
  cameraButton: $("#cameraButton"),
  replaceButton: $("#replaceButton"),
  resultPanel: $("#resultPanel"),
  resultEmpty: $("#resultEmpty"),
  resultLoading: $("#resultLoading"),
  resultError: $("#resultError"),
  resultContent: $("#resultContent"),
  loadingTitle: $("#loadingTitle"),
  loadingHint: $("#loadingHint"),
  progressBar: $("#progressBar"),
  errorMessage: $("#errorMessage"),
  retryButton: $("#retryButton"),
  summaryNumber: $("#summaryNumber"),
  summaryChip: $("#summaryChip"),
  plateList: $("#plateList"),
  annotatedImage: $("#annotatedImage"),
  downloadButton: $("#downloadButton"),
  resultTime: $("#resultTime"),
  uploadLimit: $("#uploadLimit"),
  keyDialog: $("#keyDialog"),
  keyInput: $("#keyInput"),
  keyForm: $("#keyForm"),
  keyCancel: $("#keyCancel"),
};

const state = {
  file: null,
  previewUrl: "",
  accessKey: localStorage.getItem("lpr_access_key") || "",
  info: null,
  busy: false,
  progressTimers: [],
  lastResult: null,
  infoRetry: null,
};

function setModelState(kind, text) {
  elements.modelPill.classList.remove("is-ready", "is-error");
  if (kind === "ready") elements.modelPill.classList.add("is-ready");
  if (kind === "error") elements.modelPill.classList.add("is-error");
  elements.modelStatus.textContent = text;
}

async function loadInfo() {
  clearTimeout(state.infoRetry);
  try {
    const response = await fetch("/api/info", { cache: "no-store" });
    if (!response.ok) throw new Error("模型状态不可用");
    const info = await response.json();
    state.info = info;
    elements.uploadLimit.textContent = `单张最大 ${info.max_upload_mb} MB`;
    if (info.ready) {
      setModelState("ready", "模型在线 · 可以识别");
    } else if (info.error) {
      setModelState("error", "模型加载失败");
    } else {
      setModelState("loading", "模型正在预热");
      state.infoRetry = setTimeout(loadInfo, 3000);
    }
  } catch (error) {
    setModelState("error", "无法连接服务");
    state.infoRetry = setTimeout(loadInfo, 5000);
  }
}

function isImageFile(file) {
  if (!file) return false;
  if (file.type && file.type.startsWith("image/")) return true;
  return /\.(jpe?g|png|webp)$/i.test(file.name || "");
}

function clearPreviewUrl() {
  if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
  state.previewUrl = "";
}

function setBusy(value) {
  state.busy = value;
  elements.dropzone.classList.toggle("is-busy", value);
  elements.pickButton.disabled = value;
  elements.cameraButton.disabled = value;
  elements.replaceButton.disabled = value;
  elements.retryButton.disabled = value;
}

function showView(name) {
  for (const [key, element] of Object.entries({
    empty: elements.resultEmpty,
    loading: elements.resultLoading,
    error: elements.resultError,
    content: elements.resultContent,
  })) {
    element.classList.toggle("is-hidden", key !== name);
  }
}

function startProgress() {
  stopProgress();
  const steps = [
    [0, "8%", "正在定位车牌", "YOLO 检测模型正在分析画面"],
    [350, "32%", "正在校正车牌", "分析颜色与倾斜角度"],
    [1200, "62%", "正在识别字符", "PP-OCRv4 读取车牌文本"],
    [3500, "84%", "正在融合候选", "校验格式并计算置信度"],
  ];
  for (const [delay, width, title, hint] of steps) {
    const timer = setTimeout(() => {
      elements.progressBar.style.width = width;
      elements.loadingTitle.textContent = title;
      elements.loadingHint.textContent = hint;
    }, delay);
    state.progressTimers.push(timer);
  }
}

function stopProgress() {
  state.progressTimers.forEach(clearTimeout);
  state.progressTimers = [];
  elements.progressBar.style.width = "8%";
}

function toast(message) {
  let node = document.querySelector(".toast");
  if (!node) {
    node = document.createElement("div");
    node.className = "toast";
    document.body.appendChild(node);
  }
  node.textContent = message;
  requestAnimationFrame(() => node.classList.add("is-visible"));
  clearTimeout(node._timer);
  node._timer = setTimeout(() => node.classList.remove("is-visible"), 2200);
}

function requestAccessKey() {
  return new Promise((resolve) => {
    if (!elements.keyDialog || typeof elements.keyDialog.showModal !== "function") {
      const value = window.prompt("请输入访问码") || "";
      resolve(value.trim());
      return;
    }
    elements.keyInput.value = state.accessKey || "";
    elements.keyDialog.returnValue = "";
    elements.keyDialog.showModal();
    elements.keyInput.focus();
    const onClose = () => {
      elements.keyDialog.removeEventListener("close", onClose);
      const key = elements.keyDialog.returnValue === "confirm"
        ? elements.keyInput.value.trim()
        : "";
      if (key) {
        state.accessKey = key;
        localStorage.setItem("lpr_access_key", key);
      }
      resolve(key);
    };
    elements.keyDialog.addEventListener("close", onClose);
  });
}

async function analyze(options = {}) {
  if (!state.file || state.busy) return;
  const retryAuth = options.retryAuth !== false;

  if (state.info?.auth_required && !state.accessKey) {
    const key = await requestAccessKey();
    if (!key) {
      elements.resultTime.textContent = "需要访问码";
      return;
    }
  }

  setBusy(true);
  showView("loading");
  elements.resultTime.textContent = "分析中";
  startProgress();

  const form = new FormData();
  form.append("file", state.file, state.file.name || "plate.jpg");
  const headers = {};
  if (state.accessKey) headers["X-Access-Key"] = state.accessKey;

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 120000);

  try {
    const response = await fetch("/api/recognize", {
      method: "POST",
      body: form,
      headers,
      signal: controller.signal,
    });
    const payload = await response.json().catch(() => ({}));

    if (response.status === 401) {
      state.accessKey = "";
      localStorage.removeItem("lpr_access_key");
      if (retryAuth) {
        const key = await requestAccessKey();
        if (key) return analyze({ retryAuth: false });
      }
      throw new Error("访问码无效或缺失");
    }
    if (!response.ok) {
      throw new Error(payload.detail || `服务返回错误 (${response.status})`);
    }

    renderResult(payload);
  } catch (error) {
    const message = error.name === "AbortError"
      ? "处理超时，请换一张更小的图片重试"
      : (error.message || "识别失败，请稍后重试");
    showError(message);
  } finally {
    clearTimeout(timeout);
    stopProgress();
    setBusy(false);
  }
}

function addMeta(parent, text, className = "") {
  const item = document.createElement("span");
  if (className) item.classList.add(className);
  item.textContent = text;
  parent.appendChild(item);
}

function colorLabel(color) {
  return ({ blue: "蓝牌", green: "绿牌", yellow: "黄牌", unknown: "颜色未知" })[color] || "颜色未知";
}

async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
  } catch (error) {
    const area = document.createElement("textarea");
    area.value = text;
    area.style.position = "fixed";
    area.style.opacity = "0";
    document.body.appendChild(area);
    area.select();
    document.execCommand("copy");
    area.remove();
  }
  toast(`已复制 ${text}`);
}

function plateCard(plate, index) {
  const card = document.createElement("article");
  card.className = `plate-card${plate.accepted ? "" : " is-rejected"}`;
  card.style.animationDelay = `${index * 55}ms`;

  const top = document.createElement("div");
  top.className = "plate-top";
  const text = document.createElement("div");
  text.className = "plate-text";
  text.textContent = plate.accepted ? plate.text : "无法可靠识别";

  if (plate.accepted) {
    const copy = document.createElement("button");
    copy.type = "button";
    copy.className = "copy-btn";
    copy.setAttribute("aria-label", "复制车牌号码");
    copy.innerHTML = '<svg viewBox="0 0 20 20" fill="none" aria-hidden="true"><rect x="6.5" y="6.5" width="9" height="9" rx="1.5" stroke="currentColor" stroke-width="1.4"/><path d="M13.2 6.5V5.8A1.8 1.8 0 0 0 11.4 4H5.8A1.8 1.8 0 0 0 4 5.8v5.6a1.8 1.8 0 0 0 1.8 1.8h.7" stroke="currentColor" stroke-width="1.4"/></svg>';
    copy.addEventListener("click", () => copyText(plate.text));
    top.append(text, copy);
  } else {
    top.appendChild(text);
  }
  card.appendChild(top);

  const meta = document.createElement("div");
  meta.className = "plate-meta";
  const color = document.createElement("span");
  const swatch = document.createElement("i");
  swatch.className = `color-swatch color-${plate.plate_color || "unknown"}`;
  color.append(swatch, document.createTextNode(colorLabel(plate.plate_color)));
  meta.appendChild(color);
  addMeta(meta, `综合 ${(plate.score * 100).toFixed(1)}%`);
  addMeta(meta, `检测 ${(plate.detection_confidence * 100).toFixed(0)}%`);
  if (plate.ocr_confidence > 0) addMeta(meta, `OCR ${(plate.ocr_confidence * 100).toFixed(1)}%`);
  addMeta(meta, `${plate.processing_time_ms.toFixed(0)} ms`);
  if (!plate.accepted && plate.best_text) addMeta(meta, `最高候选 ${plate.best_text}`);
  card.appendChild(meta);

  if (!plate.accepted && plate.error) {
    const error = document.createElement("p");
    error.className = "plate-error";
    error.textContent = plate.error === "low_confidence" ? "置信度低于可靠输出阈值" : "候选文本未通过车牌格式校验";
    card.appendChild(error);
  }
  return card;
}

function renderResult(data) {
  state.lastResult = data;
  showView("content");
  const plates = Array.isArray(data.plates) ? data.plates : [];
  const accepted = plates.filter((item) => item.accepted).length;
  elements.summaryNumber.textContent = String(plates.length);
  elements.summaryChip.textContent = accepted ? "识别完成" : "未得到可靠结果";
  elements.summaryChip.classList.toggle("is-warn", accepted === 0);
  elements.plateList.replaceChildren();

  if (!plates.length) {
    const empty = document.createElement("div");
    empty.className = "plate-card is-rejected";
    const title = document.createElement("div");
    title.className = "plate-text";
    title.textContent = "未检测到车牌";
    const hint = document.createElement("p");
    hint.className = "plate-error";
    hint.textContent = "请让车牌占据更大画面，并保持正面、清晰、光线充足。";
    empty.append(title, hint);
    elements.plateList.appendChild(empty);
  } else {
    plates.forEach((plate, index) => elements.plateList.appendChild(plateCard(plate, index)));
  }

  elements.annotatedImage.src = data.annotated_image || "";
  elements.downloadButton.href = data.annotated_image || "#";
  const base = (state.file?.name || "plate").replace(/\.[^.]+$/, "");
  elements.downloadButton.download = `${base}-result.jpg`;
  elements.resultTime.textContent = `${Math.round(data.elapsed_ms)} ms`;

  if (window.innerWidth <= 1040) {
    setTimeout(() => elements.resultPanel.scrollIntoView({ behavior: "smooth", block: "start" }), 120);
  }
}

function showError(message) {
  showView("error");
  elements.errorMessage.textContent = message;
  elements.resultTime.textContent = "处理失败";
}

function handleFile(file) {
  if (!isImageFile(file)) {
    toast("请选择 JPG、PNG 或 WebP 图片");
    return;
  }
  const maxMb = state.info?.max_upload_mb || 15;
  if (file.size > maxMb * 1024 * 1024) {
    toast(`图片不能超过 ${maxMb} MB`);
    return;
  }

  clearPreviewUrl();
  state.file = file;
  state.previewUrl = URL.createObjectURL(file);
  elements.previewImage.src = state.previewUrl;
  elements.previewName.textContent = file.name || "camera.jpg";
  elements.dropIdle.classList.add("is-hidden");
  elements.previewStage.classList.remove("is-hidden");
  showView("empty");
  analyze();
}

async function loadDemo(button) {
  if (state.busy) return;
  const original = button.textContent;
  button.textContent = "载入中…";
  button.disabled = true;
  try {
    const url = button.dataset.demo;
    const response = await fetch(url, { cache: "no-store" });
    if (!response.ok) throw new Error("示例图片不可用");
    const blob = await response.blob();
    const name = url.split("/").pop() || "demo.jpg";
    handleFile(new File([blob], name, { type: blob.type || "image/jpeg" }));
  } catch (error) {
    toast(error.message || "载入示例失败");
  } finally {
    button.textContent = original;
    button.disabled = false;
  }
}

function openPicker() {
  if (!state.busy) elements.fileInput.click();
}

function bindEvents() {
  elements.pickButton.addEventListener("click", (event) => {
    event.stopPropagation();
    openPicker();
  });
  elements.cameraButton.addEventListener("click", (event) => {
    event.stopPropagation();
    if (!state.busy) elements.cameraInput.click();
  });
  elements.replaceButton.addEventListener("click", (event) => {
    event.stopPropagation();
    openPicker();
  });
  elements.retryButton.addEventListener("click", () => analyze());
  elements.dropzone.addEventListener("click", (event) => {
    if (event.target.closest("button")) return;
    if (state.file && !state.busy) return;
    openPicker();
  });
  elements.dropzone.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      openPicker();
    }
  });

  for (const input of [elements.fileInput, elements.cameraInput]) {
    input.addEventListener("change", () => {
      const [file] = input.files || [];
      if (file) handleFile(file);
      input.value = "";
    });
  }

  ["dragenter", "dragover"].forEach((name) => {
    elements.dropzone.addEventListener(name, (event) => {
      event.preventDefault();
      if (!state.busy) elements.dropzone.classList.add("is-dragging");
    });
  });
  ["dragleave", "drop"].forEach((name) => {
    elements.dropzone.addEventListener(name, (event) => {
      event.preventDefault();
      elements.dropzone.classList.remove("is-dragging");
    });
  });
  elements.dropzone.addEventListener("drop", (event) => {
    const [file] = event.dataTransfer?.files || [];
    if (file && !state.busy) handleFile(file);
  });

  document.querySelectorAll("[data-demo]").forEach((button) => {
    button.addEventListener("click", () => loadDemo(button));
  });

  document.addEventListener("paste", (event) => {
    const item = Array.from(event.clipboardData?.items || []).find((entry) => entry.type.startsWith("image/"));
    if (!item) return;
    const file = item.getAsFile();
    if (file) handleFile(file);
  });

  elements.keyDialog.addEventListener("click", (event) => {
    if (event.target === elements.keyDialog) elements.keyDialog.close("cancel");
  });
  elements.keyCancel.addEventListener("click", () => elements.keyDialog.close("cancel"));

  window.addEventListener("pagehide", clearPreviewUrl);
}

function consumeUrlKey() {
  const url = new URL(window.location.href);
  const key = url.searchParams.get("key");
  if (!key) return;
  state.accessKey = key.trim();
  localStorage.setItem("lpr_access_key", state.accessKey);
  url.searchParams.delete("key");
  window.history.replaceState({}, "", `${url.pathname}${url.search}${url.hash}`);
}

async function init() {
  consumeUrlKey();
  bindEvents();
  await loadInfo();
}

init();
