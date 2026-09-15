import {memo} from 'react';
import Markdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

/** Generated text is untrusted: no raw HTML or automatic remote image requests. */
const MonarchMarkdown = memo(function MonarchMarkdown({text}: {text: string}) {
  return (
    <div className="markdown-response">
      <Markdown
        remarkPlugins={[remarkGfm]}
        skipHtml
        components={{
          a: ({href, children}) => <a href={href} target="_blank" rel="noopener noreferrer">{children}</a>,
          img: ({alt}) => <span className="markdown-image-alt">{alt || 'Image'}</span>,
          table: ({children}) => <div className="markdown-table"><table>{children}</table></div>,
        }}
      >
        {text}
      </Markdown>
    </div>
  );
});

export default MonarchMarkdown;
