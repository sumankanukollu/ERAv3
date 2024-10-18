document.addEventListener('DOMContentLoaded', function() {
  const classifyButton = document.getElementById('classifyButton');
  const urlInput = document.getElementById('urlInput');
  const resultDiv = document.getElementById('result');
  const sampleUrls = document.querySelectorAll('.sample-url');

  classifyButton.addEventListener('click', function() {
    const url = urlInput.value;
    if (url) {
      classifyUrl(url);
    } else {
      resultDiv.textContent = 'Please enter a URL';
    }
  });

  sampleUrls.forEach(sampleUrl => {
    sampleUrl.addEventListener('click', function(e) {
      e.preventDefault();
      urlInput.value = this.textContent;
    });
  });
});

async function classifyUrl(url) {
  const resultDiv = document.getElementById('result');
  resultDiv.textContent = 'Classifying...';

  try {
    const response = await fetch('http://127.0.0.1:5000/classify', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ url: url }),
    });

    if (!response.ok) {
      throw new Error('Network response was not ok');
    }

    const data = await response.json();
    displayClassification(data.classification);
  } catch (error) {
    console.error('Error:', error);
    resultDiv.textContent = 'Error classifying URL. Please try again.';
  }
}

function displayClassification(classification) {
  const resultDiv = document.getElementById('result');
  resultDiv.innerHTML = `<h3 style="font-family: Arial, sans-serif; color: #4CAF50; text-align: center;">Classification Result:</h3>
                         <p style="font-family: Arial, sans-serif; font-size: 16px; color: #333; padding: 10px; background-color: #f9f9f9; border: 1px solid #ddd; border-radius: 5px; text-align: center;">
                            ${classification}
                         </p>`;
}

