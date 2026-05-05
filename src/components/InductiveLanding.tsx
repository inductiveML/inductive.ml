import SlashMark from './SlashMark';

const V9_AREAS = [
  {
    title: 'Adapter Geometry',
    body: 'Canonical representations for LoRA adapters, seed-invariant structure, and task adaptation.',
  },
  {
    title: 'Representation Learning',
    body: 'Architectures that use learned concept slots, routing, and controlled injection into frozen decoders.',
  },
  {
    title: 'Small-Data Generalization',
    body: 'Tool-use and structured behavior learned from sparse input–output examples.',
  },
  {
    title: 'Efficient Inference',
    body: 'Sparse kernels, dynamic precision routing, and practical systems for local and frontier-scale models.',
  },
];

const V9_NOTE = {
  date: 'Apr 2026',
  title: 'Teaching Trees to Understand Geometry',
  href: '/staged-learned-coordinates',
};

const V9_PROFILE: { k: string; v: string; href?: string }[] = [
  { k: 'Email', v: 'aditya@inductive.ml', href: 'mailto:aditya@inductive.ml' },
  { k: 'X', v: 'x.com/inductive_ml', href: 'https://x.com/inductive_ml' },
  { k: 'GitHub', v: 'github.com/inductiveML', href: 'https://github.com/inductiveML' },
  { k: 'Hugging Face', v: 'hf.co/inductiveML', href: 'https://hf.co/inductiveML' },
  { k: 'OpenReview', v: '~Aditya_Veer_Parmar1', href: 'https://openreview.net/profile?id=~Aditya_Veer_Parmar1' },
  { k: 'ORCID', v: '0009-0000-3060-4372', href: 'https://orcid.org/0009-0000-3060-4372' },
];

export default function InductiveLanding() {
  return (
    <div className="v9">
      <div className="pad">
        <div className="topline">
          <div className="brand">
            <SlashMark size={22} />
            <span>Inductive.ML</span>
          </div>
          <div className="meta">Independent AI Research Lab — Bangalore, India — Est. MMXXVI</div>
          <div className="links">
            <a href="#research">Research</a>
            <a href="#writing">Writing</a>
            <a href="#about">About</a>
          </div>
        </div>

        <div className="grid12">
          <div className="hero-mark">
            <SlashMark size={120} />
          </div>
          <h1 className="h1">
            Inductive
            <br />
            <span className="slash">/</span>
            <span className="accent">ML.</span>
          </h1>
          <div className="h1-id">
            <div className="row">
              <span>// id</span>
              <span>v0.1</span>
            </div>
            <div className="row">
              <span>// status</span>
              <span>operating</span>
            </div>
            <div className="row">
              <span>// founded</span>
              <span>2026</span>
            </div>
            <div className="row">
              <span>// areas</span>
              <span>04</span>
            </div>
            <div className="row">
              <span>// notes</span>
              <span>01</span>
            </div>
            <div className="row">
              <span>// loc</span>
              <span>blr · in</span>
            </div>
          </div>

          <div className="lede-row">
            <div className="tag">— Premise</div>
            <p className="lede">
              Independent research on <span className="em">representation</span>, adaptation, and intelligence.
            </p>
            <p className="body">
              An independent AI research lab studying how learned systems represent, adapt, compress, and
              generalize.
            </p>
          </div>

          <div className="secbar" id="research">
            <span className="num">§ 01</span>
            <span className="ttl">// research</span>
            <span className="meta">04 active areas</span>
          </div>
          <div className="areas">
            {V9_AREAS.map((a, i) => (
              <div className="cell" key={a.title}>
                <div className="meta">
                  <span className="idx">[{String(i + 1).padStart(2, '0')}]</span>
                </div>
                <h3>{a.title}</h3>
                <p>{a.body}</p>
              </div>
            ))}
          </div>

          <div className="secbar" id="writing">
            <span className="num">§ 02</span>
            <span className="ttl">// writing</span>
            <span className="meta">01 latest</span>
          </div>
          <a className="write" href={V9_NOTE.href}>
            <div className="idx">[01]</div>
            <div className="date">{V9_NOTE.date}</div>
            <div className="ttl">{V9_NOTE.title}</div>
            <div className="arr">Read →</div>
          </a>

          <div className="secbar">
            <span className="num">§ 03</span>
            <span className="ttl">// mission</span>
            <span className="meta">statement</span>
          </div>
          <div className="miss">
            <div className="lbl">// purpose</div>
            <p>
              To do <strong>rigorous, independent</strong> AI research outside traditional institutional walls
              — small, sharp bets on representation geometry, efficient adaptation, and systems that reveal
              how intelligence is organized.
            </p>
            <div className="signature">— A. V. Parmar · Founder · MMXXVI</div>
          </div>

          <div className="secbar" id="about">
            <span className="num">§ 04</span>
            <span className="ttl">// about</span>
            <span className="meta">founder · principal</span>
          </div>
          <div className="about">
            <div className="name-col">
              <div className="name">
                Aditya Veer
                <br />
                Parmar
              </div>
              <div className="role">
                <span>// role</span>
                <span>Principal</span>
                <span>// loc</span>
                <span>Bangalore</span>
                <span>// since</span>
                <span>2026</span>
              </div>
            </div>
            <div className="bio">
              <p>
                Aditya Veer Parmar is an independent ML researcher and software engineer based in Bangalore,
                India. His work focuses on representation learning, LoRA geometry, LLM adaptation,
                interpretability, and AI systems.
              </p>
              <p>
                Inductive ML is run as a single-author lab — designing experiments, writing the code,
                publishing the notes.
              </p>
            </div>
            <div className="links">
              <ul>
                {V9_PROFILE.map((p) => (
                  <li key={p.k}>
                    <span className="k">{p.k}</span>
                    <span className="v">
                      {p.href ? (
                        <a href={p.href} target={p.href.startsWith('http') ? '_blank' : undefined} rel="noreferrer">
                          {p.v}
                        </a>
                      ) : (
                        p.v
                      )}
                    </span>
                  </li>
                ))}
              </ul>
            </div>
          </div>

          <div className="foot">
            <div className="a">© MMXXVI · Inductive.ML</div>
            <div className="b">Bangalore — India</div>
            <div className="c">v0.1 · inductive.ml</div>
          </div>
        </div>
      </div>
    </div>
  );
}
