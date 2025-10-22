// sketch.js

let population = [];
let numOrganisms = 200;
let generation = 0;

// --- UI Elements ---
let sliderEnvHue, sliderEnvSize, sliderMutation;
let labelEnvHue, labelEnvSize, labelMutation;
let btnRestart;

function setup() {
  createCanvas(700, 700);
  colorMode(HSB, 360, 100, 100);
  
  // --- Create all the UI elements ---
  createUI();
  
  // --- Create the initial population ---
  initializePopulation();

  frameRate(10); // Run 10 generations per second
}

// --- Initialize a new random population ---
function initializePopulation() {
  population = [];
  generation = 0;
  for (let i = 0; i < numOrganisms; i++) {
    // Create an organism with random genes
    let x = random(width);
    let y = random(height);
    let hue = random(360);
    let size = random(5, 30);
    population.push(new Organism(x, y, hue, size));
  }
}

// --- Create Sliders and Buttons ---
function createUI() {
  // Get the div we made in index.html
  let controlsDiv = select('#controls-container');

  // --- Target Hue Slider ---
  let hueDiv = createDiv().addClass('control-item').parent(controlsDiv);
  labelEnvHue = createP('Target Color (Hue): 120').parent(hueDiv);
  sliderEnvHue = createSlider(0, 360, 120).parent(hueDiv);
  sliderEnvHue.input(() => { // Update label text when slider moves
    labelEnvHue.html('Target Color (Hue): ' + sliderEnvHue.value());
  });
  
  // --- Target Size Slider ---
  let sizeDiv = createDiv().addClass('control-item').parent(controlsDiv);
  labelEnvSize = createP('Target Size: 15').parent(sizeDiv);
  sliderEnvSize = createSlider(5, 30, 15).parent(sizeDiv);
  sliderEnvSize.input(() => {
    labelEnvSize.html('Target Size: ' + sliderEnvSize.value());
  });
  
  // --- Mutation Rate Slider ---
  let mutDiv = createDiv().addClass('control-item').parent(controlsDiv);
  labelMutation = createP('Mutation Rate: 2').parent(mutDiv);
  sliderMutation = createSlider(0, 10, 2).parent(mutDiv);
  sliderMutation.input(() => {
    labelMutation.html('Mutation Rate: ' + sliderMutation.value());
  });

  // --- Restart Button ---
  let btnDiv = createDiv().addClass('control-item').parent(controlsDiv);
  btnRestart = createButton('Restart Population').parent(btnDiv);
  btnRestart.mousePressed(initializePopulation); // Call this function when clicked
}


function draw() {
  background(0, 0, 10); // Dark background

  // --- Get current values from the sliders ---
  let targetHue = sliderEnvHue.value();
  let targetSize = sliderEnvSize.value();
  let mutationRate = sliderMutation.value();
  
  // --- Run the simulation loop ---
  
  // 1. Calculate fitness for every organism
  let maxFitness = 0;
  for (let org of population) {
    org.calculateFitness(targetHue, targetSize);
    if (org.fitness > maxFitness) {
      maxFitness = org.fitness;
    }
  }

  // 2. Create the "mating pool"
  let matingPool = [];
  for (let org of population) {
    // Normalize fitness (0 to 1)
    let fitness = map(org.fitness, 0, maxFitness, 0, 1);
    // Add to pool based on fitness
    let n = floor(fitness * 100); 
    for (let j = 0; j < n; j++) {
      matingPool.push(org);
    }
  }

  // 3. Create the new generation
  let newPopulation = [];
  for (let i = 0; i < numOrganisms; i++) {
    // Pick one parent from the mating pool
    // If the pool is empty (everyone died), pick from the old pop
    let parent = random(matingPool.length > 0 ? matingPool : population);
    
    // Create an offspring
    let offspring = parent.reproduce(mutationRate);
    newPopulation.push(offspring);
  }

  // 4. Replace the old population
  population = newPopulation;
  generation++;

  // 5. Display the population
  for (let org of population) {
    org.display();
  }
  
  // --- Display Info ---
  fill(255);
  noStroke();
  textSize(24);
  text("Generation: " + generation, 10, 30);
  
  // Draw a "target" visual
  fill(targetHue, 90, 90);
  stroke(255);
  strokeWeight(2);
  ellipse(width - 40, 40, targetSize, targetSize);
}


// ===================================
//  ORGANISM CLASS
// ===================================
class Organism {
  constructor(x, y, hue, size) {
    this.pos = createVector(x, y); // Position
    // This is the "Genotype"
    this.genes = {
      hue: hue,
      size: size
    };
    this.fitness = 0;
  }

  // Calculate fitness based on matching the environment
  calculateFitness(targetHue, targetSize) {
    // Calculate fitness for HUE (0-1)
    let hueDist = abs(this.genes.hue - targetHue);
    let hueDistWrapped = min(hueDist, 360 - hueDist); // Handle 360-degree circle
    let hueFitness = map(hueDistWrapped, 0, 180, 1, 0); // 0 dist = 1 fitness

    // Calculate fitness for SIZE (0-1)
    let sizeDist = abs(this.genes.size - targetSize);
    let sizeFitness = map(sizeDist, 0, 25, 1, 0); // 0 dist = 1 fitness
    sizeFitness = constrain(sizeFitness, 0, 1);

    // Total fitness is the product of individual trait fitness
    // This means they MUST be good at BOTH to survive
    this.fitness = hueFitness * sizeFitness;
    
    // Squaring it makes selection stronger (optional, but effective)
    this.fitness = pow(this.fitness, 2);
  }

  // Create a new offspring
  reproduce(mutationRate) {
    // Start with parent's genes
    let newHue = this.genes.hue;
    let newSize = this.genes.size;

    // --- Mutate Hue ---
    newHue += random(-mutationRate, mutationRate);
    // Handle "wrap-around" for hue
    if (newHue > 360) newHue -= 360;
    if (newHue < 0) newHue += 360;
    
    // --- Mutate Size ---
    newSize += random(-mutationRate, mutationRate);
    // Constrain size to min/max
    newSize = constrain(newSize, 5, 30);

    // Offspring appears at a random new position
    let x = random(width);
    let y = random(height);
    
    return new Organism(x, y, newHue, newSize);
  }

  // Draw the organism
  display() {
    noStroke();
    // Color is set by its hue gene
    fill(this.genes.hue, 90, 90);
    // Size is set by its size gene
    ellipse(this.pos.x, this.pos.y, this.genes.size, this.genes.size);
  }
}