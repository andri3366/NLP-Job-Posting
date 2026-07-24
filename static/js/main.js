$(document).ready(function() {
    // Handle Get Explanation button click
    $('#getExplanation').click(function() {
        const prompt = $('#explanationPrompt').val();
        const mode = $(this).data('mode');
        
        if (!prompt.trim()) {
            alert('Please enter a question first.');
            return;
        }
        
        // Show loading state
        $(this).html('<span class="spinner-border spinner-border-sm" role="status"></span> Generating...');
        $(this).prop('disabled', true);
        
        // Make AJAX request
        $.ajax({
            url: '/get_explanation',
            method: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({
                prompt: prompt,
                mode: mode
            }),
            success: function(response) {
                if (response.explanation) {
                    $('#explanationResult').html(`
                        <div class="alert alert-info">
                            <i class="fas fa-lightbulb"></i> 
                            <strong>Explanation:</strong> ${response.explanation}
                        </div>
                    `);
                } else if (response.error) {
                    $('#explanationResult').html(`
                        <div class="alert alert-danger">
                            <i class="fas fa-exclamation-circle"></i> 
                            ${response.error}
                        </div>
                    `);
                }
            },
            error: function(xhr, status, error) {
                $('#explanationResult').html(`
                    <div class="alert alert-danger">
                        <i class="fas fa-exclamation-circle"></i> 
                        Error generating explanation. Please try again.
                    </div>
                `);
                console.error('Error:', error);
            },
            complete: function() {
                // Reset button state
                $('#getExplanation').html('<i class="fas fa-comment"></i> Get Explanation');
                $('#getExplanation').prop('disabled', false);
            }
        });
    });
    
    // Auto-scroll to results when form is submitted
    if (window.location.search.includes('result')) {
        $('html, body').animate({
            scrollTop: $('.card').offset().top - 100
        }, 500);
    }
    
    // Form validation
    $('form').submit(function(e) {
        const textarea = $(this).find('textarea[name="text"]');
        if (textarea.length && !textarea.val().trim()) {
            e.preventDefault();
            alert('Please enter a job description.');
            textarea.focus();
        }
    });
});

// Enable tooltips
document.addEventListener('DOMContentLoaded', function() {
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function(tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
});