function showImage(beladiri) {
    document.getElementById('selectedImage').innerHTML = `<h2>${beladiri}</h2><img src="${beladiri.toLowerCase()}.jpg" alt="${beladiri}">`;
}