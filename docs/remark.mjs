import { existsSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const pages = path.join(here, 'docs');

function walk(node, f) {
	f(node);
	node.children?.forEach((c) => walk(c, f));
}

// ```python source blocks list names from src/flodr, or example scripts; the site shows
// their signatures, or the scripts.
// Relative .md links, which GitHub follows, become site routes.
export default function remarkFlodr({ base, sigs }) {
	return (tree, file) => {
		const dir = path.relative(pages, path.dirname(file.path));
		walk(tree, (node) => {
			if (node.type === 'code' && node.meta === 'source') {
				node.value = node.value
					.split('\n')
					.map((name) => {
						if (!(name in sigs)) throw new Error(`${file.path}: ${name} is neither in src/flodr nor an example`);
						return sigs[name];
					})
					.join('\n');
				node.meta = null;
			}
			if (node.type === 'link' && /^[^:#]+\.md(#|$)/.test(node.url)) {
				const [p, hash] = node.url.split('#');
				if (!existsSync(path.join(path.dirname(file.path), p))) throw new Error(`${file.path}: broken link ${node.url}`);
				const route = path.posix.join(dir, p).replace(/\.md$/, '').replace(/(^|\/)index$/, '');
				node.url = `${base}/${route ? route + '/' : ''}${hash ? '#' + hash : ''}`;
			}
		});
	};
}
