(function () {
  const segmentationForm = document.getElementById('segmentation-form');
  const segmentationResults = document.getElementById('segmentation-results');
  const segmentationOriginal = document.getElementById('segmentation-original');
  const segmentationMask = document.getElementById('segmentation-mask');
  const segmentationOverlay = document.getElementById('segmentation-overlay');
  const segmentationSubmit = document.getElementById('segmentation-submit');
  const segmentationLoader = document.getElementById('segmentation-loader');

  const augmentationForm = document.getElementById('augmentation-form');
  const augmentationResults = document.getElementById('augmentation-results');
  const augmentationSubmit = document.getElementById('augmentation-submit');
  const augmentationLoader = document.getElementById('augmentation-loader');

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

  function setLoading(button, loader, isLoading) {
    if (button) {
      button.disabled = isLoading;
      button.setAttribute('aria-busy', String(isLoading));
    }
    if (loader) {
      loader.classList.toggle('d-none', !isLoading);
    }
  }

  function createImageCard(title, src, alt) {
    const col = document.createElement('div');
    col.className = 'col-md-4 col-sm-6';
    col.innerHTML = `
      <div class="card h-100">
        <div class="card-header">${title}</div>
        <img class="card-img-top" src="${src}" alt="${alt}" />
      </div>
    `;
    return col;
  }

  segmentationForm?.addEventListener('submit', async (event) => {
    event.preventDefault();
    hideError();
    setLoading(segmentationSubmit, segmentationLoader, true);
    try {
      const payload = await submitForm(segmentationForm, '/predict');
      segmentationOriginal.src = payload.original;
      segmentationMask.src = payload.mask;
      segmentationOverlay.src = payload.overlay;
      segmentationResults.hidden = false;
    } catch (error) {
      showError(error.message);
    } finally {
      setLoading(segmentationSubmit, segmentationLoader, false);
    }
  });

  augmentationForm?.addEventListener('submit', async (event) => {
    event.preventDefault();
    hideError();
    setLoading(augmentationSubmit, augmentationLoader, true);
    try {
      const payload = await submitForm(augmentationForm, '/augment');
      augmentationResults.innerHTML = '';
      if (payload.original) {
        augmentationResults.appendChild(
          createImageCard('Original', payload.original, 'Original image')
        );
      }
      payload.augmentations.forEach((item) => {
        augmentationResults.appendChild(
          createImageCard(item.name, item.image, item.name)
        );

      });
    } catch (error) {
      showError(error.message);
    } finally {
      setLoading(augmentationSubmit, augmentationLoader, false);
    }
  });
})();