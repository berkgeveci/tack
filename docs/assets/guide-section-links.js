// Retain bookmarks when a guide section moves to another chapter.
function followGuideSection() {
  let fragment;
  try {
    fragment = decodeURIComponent(window.location.hash.slice(1));
  } catch {
    return;
  }
  if (!fragment) return;
  const anchor = document.getElementById(fragment);
  const destination = anchor && anchor.getAttribute("data-guide-forward");
  if (destination) window.location.replace(destination);
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", followGuideSection);
} else {
  followGuideSection();
}
window.addEventListener("hashchange", followGuideSection);
