document.getElementById("extractButtonimage").addEventListener('click', async () => {
    try {
        const jokeElement = document.getElementById('extracted-text2');
        const loader = document.getElementById('loader2');
        loader.style.display = 'inline-block';
        jokeElement.textContent = "Processing... Please wait.";

        const fileInput = document.getElementById('fileInput');
        const file = fileInput.files[0];

        if (!file) {
            jokeElement.textContent = "Please select a file!";
            loader.style.display = 'none';
            return;
        }

        const formData = new FormData();
        formData.append('file', file);

        let response = await fetch('http://127.0.0.1:8001/analyze/', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status} - ${await response.text()}`);
        }

        let data = await response.json();
        console.log("API Response:", data);

        loader.style.display = 'none';

        const classification = data.final_decision === "true" ? "Real Photo" : "Fake Photo";
        const truelines = data.true_count;
        const falselines = data.false_count;

        if(truelines === 0 || falselines === 0) {
            jokeElement.innerHTML = `<p>${classification}</p>`;
            return;
        }else if(truelines > falselines) {
            jokeElement.innerHTML = `<p>${classification} - False Lines: ${falselines}</p>`;
            return;
        }else if(falselines > truelines) {
            jokeElement.innerHTML = `<p>${classification} - True Lines: ${truelines}</p>`;
            return;
        }


    } catch (err) {
        console.error("Error:", err);
        document.getElementById('extracted-text').textContent = "Error fetching classification!";
    }
});


document.getElementById("extractButtontext").addEventListener('click', async () => {
    try {
        const jokeElement = document.getElementById('extracted-text1');
        const loader = document.getElementById('loader1');
        loader.style.display = 'inline-block';

        // Step 1: Show initial processing message
        jokeElement.textContent = "Processing... Please wait.";

        const article_text = document.querySelector(".text-input").value.trim(); // Trim to avoid empty spaces

        if (!article_text) { 
            jokeElement.textContent = "Please enter some text!";
            loader.style.display = 'none';
            return; // Stop execution if input is empty
        }

        let response = await fetch('http://127.0.0.1:8000/classify_news', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ article_text })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }

        let data = await response.json();
        console.log("API Response:", data);

        // Step 2: Extract classification label properly
        const classification = data.External_fact_verification || "No classification found";

        loader.style.display = 'none';

        // Step 3: Delay classification update
        
        jokeElement.innerHTML = `<p>${classification}</p>`; // ✅ Correct usage
        

    } catch (err) {
        console.error("Error:", err);
        document.getElementById('extracted-text').textContent = "Error fetching classification!";
    }
});

