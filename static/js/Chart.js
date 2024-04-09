document.getElementById('analyzeBtn').addEventListener('click', function() {
    var chartContainer = document.getElementById('chartContainer');
    var isChartVisible = chartContainer.style.display === 'block';

    // Toggle visibility of the chart container
    chartContainer.style.display = isChartVisible ? 'none' : 'block';

    if (!isChartVisible) {
        var flagsData = JSON.parse(document.getElementById('dataContainer').getAttribute('data-flags'));

        var labels = flagsData.map(function(item) { return item.Class; });
        var percentages = flagsData.map(function(item) { return item.Percentage; });

        var data = {
            datasets: [{
                data: percentages,
                backgroundColor: [
                    // Provide enough colors for each class
                    'rgba(255, 99, 132, 0.6)',
                    'rgba(54, 162, 235, 0.6)',
                    'rgba(255, 206, 86, 0.6)',
                    'rgba(75, 192, 192, 0.6)',
                    'rgba(153, 102, 255, 0.6)',
                    'rgba(255, 159, 64, 0.6)'
                ],
            }],
            labels: labels
        };

        var ctx = document.getElementById('pieChart').getContext('2d');
        new Chart(ctx, {
            type: 'pie',
            data: data,
            options: {
                responsive: true,
                maintainAspectRatio: false,
            }
        });
    }
});

