document.querySelector('.file-input').addEventListener('change', function(event) {
  const fileName = event.target.files[0].name;
  document.querySelector('.file-name').textContent = fileName;
});

document.getElementById('uploadForm').addEventListener('submit', function() {
  document.getElementById('loadingSpinner').style.display = 'block';
});
