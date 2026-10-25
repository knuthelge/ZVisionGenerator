// Move a section banner out of the content column, so it spans the full width above the navigation and the page.
document.addEventListener("DOMContentLoaded", () => {
  const banner = document.querySelector(".md-content .zb");
  const main = document.querySelector(".md-main");
  if (!banner || !main) return;
  banner.classList.add("zb--wide", "md-typeset");
  main.before(banner);
});
