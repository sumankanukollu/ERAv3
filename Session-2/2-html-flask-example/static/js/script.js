document.addEventListener('DOMContentLoaded', () => {
    const animalSelect = document.getElementById('animalSelect');
    const animalImage = document.getElementById('animalImage');
    const fileInput = document.getElementById('fileInput');
    const fileInfo = document.getElementById('fileInfo');

    animalSelect.addEventListener('change', async () => {
        const animal = animalSelect.value;
        if (animal) {
            const response = await fetch(`/get_animal_image/${animal}`);
            const imageUrl = await response.text();
            if (imageUrl) {
                animalImage.innerHTML = `<img src="${imageUrl}" alt="${animal}" style="max-width: 100%;">`;
            } else {
                animalImage.innerHTML = 'Image not found';
            }
        } else {
            animalImage.innerHTML = '';
        }
    });

    fileInput.addEventListener('change', async () => {
        const file = fileInput.files[0];
        if (file) {
            const formData = new FormData();
            formData.append('file', file);

            const response = await fetch('/upload_file', {
                method: 'POST',
                body: formData
            });

            const fileDetails = await response.json();
            fileInfo.innerHTML = `
                <p>File Name: ${fileDetails.name}</p>
                <p>File Size: ${fileDetails.size} bytes</p>
                <p>File Type: ${fileDetails.type}</p>
            `;
        } else {
            fileInfo.innerHTML = '';
        }
    });
});
