import {describe, expect, test} from 'bun:test';
import {createElement} from 'react';
import {renderToStaticMarkup} from 'react-dom/server';
import MonarchMarkdown from '../src/components/MonarchMarkdown';

const render = (text: string) => renderToStaticMarkup(createElement(MonarchMarkdown, {text}));

describe('generated response Markdown', () => {
  test('renders prose, JSON fences, GFM tables, and checklists, including partial streams', () => {
    const html = render('## Result\n\n**Ada** uses `Python`.\n\n```json\n{"name":"Ada"}\n```\n\n| Name | Age |\n| --- | --- |\n| Ada | 28 |\n\n- [x] Extracted\n- [ ] Review\n\n> A quoted note.');
    for (const tag of ['<h2>', '<strong>', '<code>', '<pre>', '<table>', '<blockquote>', 'type="checkbox"', 'class="language-json"']) expect(html).toContain(tag);
    expect(html).not.toContain('```');
    expect(render('```json\n{"name":')).toContain('class="language-json"');
    expect(render('[Reference](https://example.com)')).toContain('rel="noopener noreferrer"');
  });

  test('does not execute model HTML, unsafe links, or fetch model-supplied images', () => {
    const html = render('<script>alert(1)</script>\n\n<img src="https://example.com/tracker" onerror="alert(1)">\n\n[bad](javascript:alert%281%29)\n\n![A diagram](https://example.com/tracker)');
    expect(html).not.toContain('<script');
    expect(html).not.toContain('<img');
    expect(html).not.toContain('onerror');
    expect(html).not.toContain('javascript:');
    expect(html).not.toContain('https://example.com/tracker');
    expect(html).toContain('A diagram');
  });
});
