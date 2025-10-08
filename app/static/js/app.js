(function () {
  const segmentationForm = document.getElementById('segmentation-form');
  const segmentationResults = document.getElementById('segmentation-results');
  const segmentationOriginal = document.getElementById('segmentation-original');
  const segmentationMask = document.getElementById('segmentation-mask');
  const segmentationOverlay = document.getElementById('segmentation-overlay');

  const augmentationForm = document.getElementById('augmentation-form');
  const augmentationResults = document.getElementById('augmentation-results');

  const errorAlert = document.getElementById('error-alert');

  async function submitForm(form, endpoint) {
    const formData = new FormData(form);
    const response = await fetch(endpoint, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const payload = await response.json().catch(() => ({}));
      throw new Error(payload.error || 'Request failed');
    }

    return response.json();
  }

  function showError(message) {
    errorAlert.textContent = message;
    errorAlert.classList.remove('d-none');
  }

  function hideError() {
    errorAlert.classList.add('d-none');
    errorAlert.textContent = '';
  }

  segmentationForm?.addEventListener('submit', async (event) => {
    event.preventDefault();
    hideError();
    try {
      const payload = await submitForm(segmentationForm, '/predict');
      segmentationOriginal.src = payload.original;
      segmentationMask.src = payload.mask;
      segmentationOverlay.src = payload.overlay;
      segmentationResults.hidden = false;
    } catch (error) {
      showError(error.message);
    }
  });

  augmentationForm?.addEventListener('submit', async (event) => {
    event.preventDefault();
    hideError();
    try {
      const payload = await submitForm(augmentationForm, '/augment');
      augmentationResults.innerHTML = '';
      payload.augmentations.forEach((item) => {
        const col = document.createElement('div');
        col.className = 'col-md-4 col-sm-6';
        col.innerHTML = `
          <div class="card h-100">
            <div class="card-header">${item.name}</div>
            <img class="card-img-top" src="${item.image}" alt="${item.name}" />
          </div>
        `;
        augmentationResults.appendChild(col);
      });
    } catch (error) {
      showError(error.message);
    }
  });
})();