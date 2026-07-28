/* Samuel Alipour-fard — site interactions */
(function () {
  "use strict";

  /* ---- Theme (persisted, respects system on first visit) ---- */
  var root = document.documentElement;
  var stored = null;
  try { stored = localStorage.getItem("theme"); } catch (e) {}
  if (stored === "light" || stored === "dark") {
    root.setAttribute("data-theme", stored);
  } else if (window.matchMedia && window.matchMedia("(prefers-color-scheme: light)").matches) {
    root.setAttribute("data-theme", "light");
  }

  function bindThemeToggle() {
    var btn = document.querySelector(".theme-toggle");
    if (!btn) return;
    btn.addEventListener("click", function () {
      var next = root.getAttribute("data-theme") === "light" ? "dark" : "light";
      root.setAttribute("data-theme", next);
      try { localStorage.setItem("theme", next); } catch (e) {}
    });
  }

  /* ---- Sticky nav shadow on scroll ---- */
  function bindNav() {
    var nav = document.querySelector(".nav");
    if (!nav) return;
    var onScroll = function () {
      if (window.scrollY > 12) nav.classList.add("scrolled");
      else nav.classList.remove("scrolled");
    };
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });

    var toggle = document.querySelector(".nav-toggle");
    var links = document.querySelector(".nav-links");
    if (toggle && links) {
      toggle.addEventListener("click", function () { links.classList.toggle("open"); });
      links.addEventListener("click", function (e) {
        if (e.target.tagName === "A") links.classList.remove("open");
      });
    }
  }

  /* ---- Reveal on scroll ---- */
  function bindReveal() {
    var els = document.querySelectorAll(".reveal");
    if (!els.length) return;
    if (!("IntersectionObserver" in window)) {
      els.forEach(function (el) { el.classList.add("in"); });
      return;
    }
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting) {
          entry.target.classList.add("in");
          io.unobserve(entry.target);
        }
      });
    }, { threshold: 0.12, rootMargin: "0px 0px -8% 0px" });
    els.forEach(function (el) { io.observe(el); });
  }

  /* ---- Active section in nav (homepage) ---- */
  function bindScrollSpy() {
    var links = Array.prototype.slice.call(document.querySelectorAll('.nav-links a[href^="#"], .nav-links a[href*="#"]'));
    var map = {};
    links.forEach(function (a) {
      var hash = a.getAttribute("href").split("#")[1];
      if (hash) map[hash] = a;
    });
    var ids = Object.keys(map);
    if (!ids.length || !("IntersectionObserver" in window)) return;
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting) {
          links.forEach(function (l) { l.classList.remove("active"); });
          if (map[entry.target.id]) map[entry.target.id].classList.add("active");
        }
      });
    }, { rootMargin: "-45% 0px -50% 0px" });
    ids.forEach(function (id) {
      var sec = document.getElementById(id);
      if (sec) io.observe(sec);
    });
  }

  /* ---- Project filters ---- */
  function bindFilters() {
    var filters = document.querySelectorAll(".filter");
    if (!filters.length) return;
    var cards = document.querySelectorAll("[data-tags]");
    filters.forEach(function (f) {
      f.addEventListener("click", function () {
        filters.forEach(function (x) { x.classList.remove("active"); });
        f.classList.add("active");
        var key = f.getAttribute("data-filter");
        cards.forEach(function (c) {
          var tags = c.getAttribute("data-tags") || "";
          var show = key === "all" || tags.split(" ").indexOf(key) !== -1;
          c.style.display = show ? "" : "none";
        });
      });
    });
  }

  document.addEventListener("DOMContentLoaded", function () {
    bindThemeToggle();
    bindNav();
    bindReveal();
    bindScrollSpy();
    bindFilters();
  });
})();
