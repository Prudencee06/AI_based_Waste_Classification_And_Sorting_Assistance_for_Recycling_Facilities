// Main JavaScript for Waste Classification System

// Auto-hide alerts after 5 seconds
document.addEventListener('DOMContentLoaded', function() {
    // Add fade-out to alerts
    const alerts = document.querySelectorAll('.alert');
    alerts.forEach(alert => {
        setTimeout(() => {
            alert.style.transition = 'opacity 0.5s';
            alert.style.opacity = '0';
            setTimeout(() => alert.remove(), 500);
        }, 5000);
    });
    
    // Add tooltips to all elements with title
    const tooltips = document.querySelectorAll('[title]');
    tooltips.forEach(el => {
        new bootstrap.Tooltip(el);
    });
});

// Format date and time
function formatDateTime(dateString) {
    const date = new Date(dateString);
    return date.toLocaleString();
}

// Format duration (minutes)
function formatDuration(minutes) {
    if (minutes < 60) {
        return `${minutes} min`;
    }
    const hours = Math.floor(minutes / 60);
    const mins = minutes % 60;
    return `${hours}h ${mins}m`;
}

// Format confidence as percentage
function formatConfidence(confidence) {
    return `${(confidence * 100).toFixed(1)}%`;
}

// Get color based on material type
function getMaterialColor(material) {
    const colors = {
        'plastic': '#007bff',
        'metal': '#dc3545',
        'paper': '#28a745',
        'glass': '#17a2b8',
        'cardboard': '#ffc107',
        'organic': '#6c757d',
        'trash': '#343a40'
    };
    return colors[material] || '#6c757d';
}

// Export data to CSV
function exportToCSV(data, filename) {
    if (!data || data.length === 0) {
        console.warn('No data to export');
        return;
    }
    
    const headers = Object.keys(data[0]);
    const csvRows = [];
    
    // Add headers
    csvRows.push(headers.join(','));
    
    // Add data rows
    for (const row of data) {
        const values = headers.map(header => {
            const value = row[header] || '';
            return `"${String(value).replace(/"/g, '""')}"`;
        });
        csvRows.push(values.join(','));
    }
    
    const csv = csvRows.join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
}

// Show notification
function showNotification(message, type = 'info') {
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show position-fixed top-0 end-0 m-3`;
    alertDiv.style.zIndex = '9999';
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    document.body.appendChild(alertDiv);
    
    setTimeout(() => {
        alertDiv.remove();
    }, 5000);
}

// Validate date input
function validateDateRange(startDate, endDate) {
    if (!startDate || !endDate) {
        showNotification('Please select both start and end dates', 'warning');
        return false;
    }
    
    if (new Date(startDate) > new Date(endDate)) {
        showNotification('Start date cannot be after end date', 'warning');
        return false;
    }
    
    return true;
}

// Refresh page data
function refreshData() {
    showNotification('Refreshing data...', 'info');
    setTimeout(() => {
        window.location.reload();
    }, 500);
}

// Initialize charts helper
function initCharts() {
    // This function can be extended for chart initialization
    console.log('Charts initialized');
}