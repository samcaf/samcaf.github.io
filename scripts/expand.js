function toggleExpand(button) {
  var content = button.nextElementSibling;
  if (content.style.display === "block") {
    content.style.display = "none";
    button.innerHTML = "+";
  } else {
    content.style.display = "block";
    button.innerHTML = "-";
  }
}

var expandableElements = document.getElementsByClassName("expandable");
for (var i = 0; i < expandableElements.length; i++) {
  var button = document.createElement("button");
  button.innerHTML = "+";
  button.className = "expand-button";
  button.onclick = function() {
    toggleExpand(this);
  };
  expandableElements[i].insertBefore(button, expandableElements[i].firstChild);
  expandableElements[i].style.display = "none";
}
