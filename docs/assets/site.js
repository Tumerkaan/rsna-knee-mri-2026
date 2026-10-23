(() => {
  const root = document.documentElement;
  const themeButton = document.getElementById("theme");
  const menuButton = document.getElementById("menu");
  const nav = document.getElementById("nav");
  const applyTheme = dark => {
    root.dataset.theme = dark ? "dark" : "light";
    themeButton.textContent = dark ? "Light theme" : "Dark theme";
    themeButton.setAttribute("aria-pressed", String(dark));
  };
  let saved;
  try { saved = localStorage.getItem("rsna-theme"); } catch { saved = null; }
  applyTheme(saved ? saved === "dark" : matchMedia("(prefers-color-scheme: dark)").matches);
  themeButton.addEventListener("click", () => {
    const dark = root.dataset.theme !== "dark";
    applyTheme(dark);
    try { localStorage.setItem("rsna-theme", dark ? "dark" : "light"); } catch {}
  });
  const closeMenu = () => { nav.classList.remove("open"); menuButton.setAttribute("aria-expanded","false"); };
  menuButton.addEventListener("click", () => {
    const open = nav.classList.toggle("open");
    menuButton.setAttribute("aria-expanded",String(open));
  });
  document.addEventListener("keydown", e => { if (e.key === "Escape") closeMenu(); });
  nav.querySelectorAll("a").forEach(a => a.addEventListener("click", closeMenu));
  if ("IntersectionObserver" in window) {
    const observer = new IntersectionObserver(entries => {
      for (const entry of entries) if (entry.isIntersecting) {
        nav.querySelectorAll("a").forEach(a => a.classList.toggle("active",a.hash === "#"+entry.target.id));
      }
    }, {rootMargin:"-15% 0px -65% 0px"});
    document.querySelectorAll("section[id]").forEach(s => observer.observe(s));
  }
})();
