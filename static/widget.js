(function() {
    const IFRAME_URL = "http://127.0.0.1:5000/"; // Points to index.html via Flask root

    const style = document.createElement('style');
    style.innerHTML = `
        #skintoner-widget-container {
            position: fixed; top: 0; left: 0; width: 100%; height: 100%;
            z-index: 2147483647; background: rgba(0,0,0,0.5);
            display: none; justify-content: center; align-items: center;
            backdrop-filter: blur(5px); opacity: 0; transition: opacity 0.3s;
        }
        #skintoner-widget-container.visible { opacity: 1; }

        #skintoner-iframe {
            width: 375px; height: 800px; max-height: 90vh;
            border: none; border-radius: 30px; background: white;
            box-shadow: 0 20px 50px rgba(0,0,0,0.5);
            transition: all 0.3s ease;
        }
        @media (min-width: 768px) {
            #skintoner-iframe { width: 90vw; height: 90vh; max-width: 1200px; max-height: 900px; border-radius: 20px; }
        }
        @media (max-width: 500px) {
            #skintoner-iframe { width: 100%; height: 100%; border-radius: 0; }
        }
    `;
    document.head.appendChild(style);

    const container = document.createElement('div');
    container.id = 'skintoner-widget-container';

    // Create iframe but don't set src yet to prevent early loading
    const iframe = document.createElement('iframe');
    iframe.id = 'skintoner-iframe';
    iframe.allow = "camera; microphone";

    container.appendChild(iframe);
    document.body.appendChild(container);

    function openWidget(productId) {
        // Construct URL with product ID if provided
        let finalUrl = IFRAME_URL;
        if (productId) {
            finalUrl += `?product_id=${productId}`;
        }

        iframe.src = finalUrl;

        container.style.display = 'flex';
        setTimeout(() => container.classList.add('visible'), 10);
    }

    function closeWidget() {
        container.classList.remove('visible');
        setTimeout(() => {
            container.style.display = 'none';
            iframe.src = ""; // Reset src to stop camera
        }, 300);
    }

    window.addEventListener('message', function(event) {
        if (event.data === 'close-skintoner-widget') {
            closeWidget();
        }
    });

    window.Skintoner = { open: openWidget, close: closeWidget };
})();