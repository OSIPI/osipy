window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"], ["$", "$"]],
    processEscapes: true,
    processEnvironments: true
  },
  startup: {
    ready() {
      MathJax.startup.defaultReady();
      document$.subscribe(() => {
        MathJax.typesetClear();
        MathJax.typesetPromise();
      });
    }
  }
};
