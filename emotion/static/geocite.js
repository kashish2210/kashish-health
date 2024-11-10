// Seed-based random generator (simple example for illustration)
function randomFromSeed(seed) {
    let hash = 0;
    for (let i = 0; i < seed.length; i++) {
      hash = seed.charCodeAt(i) + ((hash << 5) - hash);
    }
    return hash;
  }
  
  // Generate random features based on seed
  function generateFeatures(seed) {
    const baseRandom = Math.abs(randomFromSeed(seed));
    document.getElementById('shapeFeature1').textContent = baseRandom % 100;
    document.getElementById('shapeFeature2').textContent = baseRandom % 1000;
    // More features can be added
  }
  
  // Event Listeners
  document.getElementById('seedInput').addEventListener('input', (e) => {
    generateFeatures(e.target.value);
  });
  
  document.getElementById('refreshButton').addEventListener('click', () => {
    const seed = document.getElementById('seedInput').value;
    generateFeatures(seed);
    renderShape(); // Function to update the display area
  });
  
  document.getElementById('erosionSlider').addEventListener('input', (e) => {
    renderShape();
  });
  
  document.getElementById('auraColor').addEventListener('input', (e) => {
    document.getElementById('displayArea').style.backgroundColor = e.target.value;
  });
  
  // Function to render a basic shape
  function renderShape() {
    const displayArea = document.getElementById('displayArea');
    const erosion = document.getElementById('erosionSlider').value;
    displayArea.innerHTML = ''; // Clear previous shapes
    const shape = document.createElement('div');
    shape.style.width = `${100 + erosion * 100}px`;
    shape.style.height = `${100 + erosion * 100}px`;
    shape.style.background = document.getElementById('auraColor').value;
    shape.style.borderRadius = '50%'; // Just an example shape
    displayArea.appendChild(shape);
  }
  
  // Initialize with default seed
  generateFeatures(document.getElementById('seedInput').value);
  renderShape();
  