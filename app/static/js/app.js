(function () {
  const segmentationForm = document.getElementById('segmentation-form');
  const segmentationResults = document.getElementById('segmentation-results');
  const segmentationOriginal = document.getElementById('segmentation-original');
  const segmentationMask = document.getElementById('segmentation-mask');
  const segmentationOverlay = document.getElementById('segmentation-overlay');
  const segmentationSubmit = document.getElementById('segmentation-submit');
  const segmentationLoader = document.getElementById('segmentation-loader');

  const augmentationRandomForm = document.getElementById('augmentation-random-form');
  const augmentationRandomResults = document.getElementById('augmentation-random-results');
  const augmentationRandomSubmit = document.getElementById('augmentation-random-submit');
  const augmentationRandomLoader = document.getElementById('augmentation-random-loader');

  const augmentationGalleryForm = document.getElementById('augmentation-gallery-form');
  const augmentationGalleryResults = document.getElementById('augmentation-gallery-results');
  const augmentationGallerySubmit = document.getElementById('augmentation-gallery-submit');
  const augmentationGalleryLoader = document.getElementById('augmentation-gallery-loader');

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

  function createImageCard(title, src, alt, columnClass = 'col-md-4 col-sm-6') {
    const col = document.createElement('div');
    col.className = columnClass;
    col.innerHTML = `
      <div class="card h-100">
        <div class="card-header">${title}</div>
        <img class="card-img-top" src="${src}" alt="${alt}" />
      </div>
    `;
    return col;
  }

  function renderAugmentationResults(container, payload, columnClass) {
    container.innerHTML = '';
    if (payload.original) {
      container.appendChild(
        createImageCard('Original', payload.original, 'Original image', columnClass)
      );
    }
    payload.augmentations.forEach((item) => {
      container.appendChild(
        createImageCard(item.name, item.image, item.name, columnClass)
      );
    });
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

  augmentationRandomForm?.addEventListener('submit', async (event) => {
    event.preventDefault();
    hideError();
    setLoading(augmentationRandomSubmit, augmentationRandomLoader, true);
    try {
      const payload = await submitForm(augmentationRandomForm, '/augment');
      renderAugmentationResults(
        augmentationRandomResults,
        payload,
        'col-md-4 col-sm-6'
      );
    } catch (error) {
      showError(error.message);
    } finally {
      setLoading(augmentationRandomSubmit, augmentationRandomLoader, false);
    }
  });

  augmentationGalleryForm?.addEventListener('submit', async (event) => {
    event.preventDefault();
    hideError();
    setLoading(augmentationGallerySubmit, augmentationGalleryLoader, true);
    try {
      const payload = await submitForm(augmentationGalleryForm, '/augment/gallery');
      renderAugmentationResults(
        augmentationGalleryResults,
        payload,
        'col-lg-3 col-md-4 col-sm-6'
      );
    } catch (error) {
      showError(error.message);
    } finally {
      setLoading(augmentationGallerySubmit, augmentationGalleryLoader, false);
    }
  });
})();