import { useEffect, useState } from "react";
import Live from "./pages/Live";
import Results from "./pages/Results";

const REPO = "https://github.com/bhatia4404/contextgc";
const PAGES = { "/": { title: "ContextGC Results", label: "Results" }, "/live": { title: "ContextGC Live Run", label: "Run it live" } };
type Path = keyof typeof PAGES;
const current = (): Path => (location.pathname.replace(/\/$/, "") === "/live" ? "/live" : "/");

export default function App() {
  const [path, setPath] = useState<Path>(current);
  useEffect(() => {
    const onPop = () => setPath(current());
    addEventListener("popstate", onPop);
    return () => removeEventListener("popstate", onPop);
  }, []);
  useEffect(() => { document.title = PAGES[path].title; }, [path]);

  const go = (to: Path) => (e: React.MouseEvent) => {
    e.preventDefault();
    history.pushState(null, "", to);
    setPath(to);
    scrollTo(0, 0);
  };

  return <>
    <header className="top">
      <span className="brand">ContextGC</span>
      <nav>
        {(Object.keys(PAGES) as Path[]).map(p =>
          <a key={p} href={p} onClick={go(p)} aria-current={path === p ? "page" : undefined}>{PAGES[p].label}</a>)}
        <a href={REPO}>Code</a>
      </nav>
    </header>
    {path === "/live" ? <Live /> : <Results />}
    <footer>ContextGC · results from GPU runs on Modal</footer>
  </>;
}
