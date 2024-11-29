// Theme Switcher
document.addEventListener('DOMContentLoaded', () => {
    const themeToggle = document.getElementById('theme-toggle');
    const themeIcon = themeToggle.querySelector('i');
    
    // Check for saved theme preference
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme) {
        document.documentElement.setAttribute('data-theme', savedTheme);
        updateThemeIcon(savedTheme === 'light');
    }

    // Toggle theme
    themeToggle.addEventListener('click', () => {
        const currentTheme = document.documentElement.getAttribute('data-theme');
        const newTheme = currentTheme === 'light' ? 'dark' : 'light';
        
        document.documentElement.setAttribute('data-theme', newTheme);
        localStorage.setItem('theme', newTheme);
        updateThemeIcon(newTheme === 'light');
    });

    function updateThemeIcon(isLight) {
        themeIcon.className = isLight ? 'fas fa-moon' : 'fas fa-sun';
    }

    // Smooth scroll for navigation links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            document.querySelector(this.getAttribute('href')).scrollIntoView({
                behavior: 'smooth'
            });
        });
    });

    // Initialize skill bars
    const skillBars = document.querySelectorAll('.skill-bar');
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const skillBar = entry.target;
                const level = skillBar.getAttribute('data-level');
                skillBar.style.setProperty('--width', `${level}%`);
                observer.unobserve(skillBar);
            }
        });
    }, { threshold: 0.5 });

    skillBars.forEach(bar => observer.observe(bar));

    // Form submission handling
    const contactForm = document.getElementById('contact-form');
    if (contactForm) {
        contactForm.addEventListener('submit', (e) => {
            e.preventDefault();
            // Add your form submission logic here
            alert('Thank you for your message! I will get back to you soon.');
            contactForm.reset();
        });
    }

    // Download CV functionality
    document.getElementById('downloadCV').addEventListener('click', function() {
        // Create a clone of the container to modify for PDF
        const container = document.querySelector('.container').cloneNode(true);
        
        // Remove elements we don't want in the PDF
        container.querySelector('.theme-toggle')?.remove();
        
        // Create PDF options
        const options = {
            margin: 10,
            filename: 'Zain_Tariq_CV.pdf',
            image: { type: 'jpeg', quality: 0.98 },
            html2canvas: { 
                scale: 2,
                useCORS: true,
                logging: false
            },
            jsPDF: { 
                unit: 'mm', 
                format: 'a4', 
                orientation: 'portrait' 
            }
        };

        // Generate PDF
        const element = document.createElement('div');
        element.appendChild(container);
        element.style.width = '100%';
        element.style.padding = '20px';
        
        // Create loading indicator
        const button = this;
        const originalText = button.innerHTML;
        button.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Generating...';
        button.disabled = true;

        // Generate and download PDF
        html2pdf().set(options).from(element).save()
            .then(() => {
                button.innerHTML = originalText;
                button.disabled = false;
            })
            .catch(err => {
                console.error('PDF generation failed:', err);
                button.innerHTML = originalText;
                button.disabled = false;
            });
    });
});

// Background animation
function animateBackground() {
    const shapes = document.querySelector('.shapes');
    if (!shapes) return;

    for (let i = 0; i < 50; i++) {
        const shape = document.createElement('div');
        shape.className = 'shape';
        shape.style.setProperty('--i', i);
        shapes.appendChild(shape);
    }
}

animateBackground();

// Add parallax effect to background shapes
document.addEventListener('mousemove', (e) => {
    const shapes = document.querySelector('.shapes');
    const x = e.clientX / window.innerWidth;
    const y = e.clientY / window.innerHeight;
    
    shapes.style.transform = `translate(${x * 20}px, ${y * 20}px)`;
});

// Add animation to project cards on scroll
const observerOptions = {
    threshold: 0.1
};

const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.style.opacity = '1';
            entry.target.style.transform = 'translateY(0)';
        }
    });
}, observerOptions);

document.querySelectorAll('.project-card').forEach(card => {
    card.style.opacity = '0';
    card.style.transform = 'translateY(20px)';
    card.style.transition = 'all 0.6s ease-out';
    observer.observe(card);
});
