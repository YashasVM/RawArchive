const state = {
  datasetId: null,
  bundleId: null,
  baseModel: "Qwen/Qwen2.5-3B-Instruct",
};

const uploadForm = document.getElementById("upload-form");
const buildForm = document.getElementById("build-form");
const registerForm = document.getElementById("register-form");

const uploadResult = document.getElementById("upload-result");
const buildResult = document.getElementById("build-result");
const colabResult = document.getElementById("colab-result");
const registerResult = document.getElementById("register-result");
const targetUserSelect = document.getElementById("target-user");

function showResult(node, content) {
  node.classList.remove("hidden");
  node.textContent = content;
}

function asJson(data) {
  return JSON.stringify(data, null, 2);
}

uploadForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const fileInput = document.getElementById("chat-file");
  const files = Array.from(fileInput.files || []);
  if (files.length === 0) {
    showResult(uploadResult, "Select one or more .json files first.");
    return;
  }

  const formData = new FormData();
  files.forEach((file) => formData.append("files", file));

  try {
    showResult(uploadResult, `Uploading and parsing ${files.length} file(s)...`);
    const response = await fetch("/v1/datasets/instagram/upload", {
      method: "POST",
      body: formData,
    });

    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Upload failed");
    }

    state.datasetId = payload.dataset_id;
    targetUserSelect.innerHTML = "";

    const placeholder = document.createElement("option");
    placeholder.value = "";
    placeholder.textContent = "Select target user";
    targetUserSelect.appendChild(placeholder);

    payload.participants.forEach((participant) => {
      const option = document.createElement("option");
      option.value = participant;
      option.textContent = participant;
      targetUserSelect.appendChild(option);
    });

    showResult(
      uploadResult,
      [
        `dataset_id: ${payload.dataset_id}`,
        `files_uploaded: ${files.length}`,
        `messages_total: ${payload.stats.messages_total}`,
        `conversations_total: ${payload.stats.conversations_total}`,
        `avg_reply_gap_sec: ${payload.stats.avg_reply_gap_sec}`,
        `warnings: ${payload.warnings.join(", ") || "none"}`,
      ].join("\n")
    );
  } catch (error) {
    showResult(uploadResult, `Error: ${error.message}`);
  }
});

buildForm.addEventListener("submit", async (event) => {
  event.preventDefault();

  if (!state.datasetId) {
    showResult(buildResult, "Upload a dataset first.");
    return;
  }

  const targetUserId = targetUserSelect.value;
  if (!targetUserId) {
    showResult(buildResult, "Select a target user.");
    return;
  }

  const body = {
    target_user_id: targetUserId,
    context_turns: Number(document.getElementById("context-turns").value),
    min_reply_chars: Number(document.getElementById("min-reply-chars").value),
    max_samples: Number(document.getElementById("max-samples").value),
    val_ratio: 0.1,
  };

  try {
    showResult(buildResult, "Building bundle...");
    const response = await fetch(`/v1/datasets/${state.datasetId}/build`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Bundle build failed");
    }

    state.bundleId = payload.bundle_id;
    showResult(
      buildResult,
      [
        `bundle_id: ${payload.bundle_id}`,
        `train_examples: ${payload.train_examples}`,
        `val_examples: ${payload.val_examples}`,
        `download_url: ${payload.download_url}`,
      ].join("\n")
    );

    const launchRes = await fetch(`/v1/colab/launch?bundle_id=${state.bundleId}`);
    const launchPayload = await launchRes.json();
    if (!launchRes.ok) {
      throw new Error(launchPayload.detail || "Launch config failed");
    }

    showResult(
      colabResult,
      [
        `notebook_url: ${launchPayload.notebook_url}`,
        `env:`,
        asJson(launchPayload.env),
      ].join("\n")
    );
  } catch (error) {
    showResult(buildResult, `Error: ${error.message}`);
  }
});

registerForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!state.bundleId) {
    showResult(registerResult, "Build a bundle before registering a model.");
    return;
  }

  const body = {
    bundle_id: state.bundleId,
    adapter_uri: document.getElementById("adapter-uri").value,
    base_model: state.baseModel,
    metrics: {
      val_loss: Number(document.getElementById("val-loss").value),
      style_score: Number(document.getElementById("style-score").value),
    },
  };

  try {
    const response = await fetch("/v1/models/register", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Model registration failed");
    }

    showResult(registerResult, `model_id: ${payload.model_id}\nstatus: ${payload.status}`);
  } catch (error) {
    showResult(registerResult, `Error: ${error.message}`);
  }
});
