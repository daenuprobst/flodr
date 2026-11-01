import starlight from '@astrojs/starlight';
import { execFileSync } from 'node:child_process';
import { readdirSync, readFileSync } from 'node:fs';
import { defineConfig } from 'astro/config';
import remarkFlodr from './remark.mjs';

const base = '/flodr';
// signatures from src/flodr and the example scripts; part of the config, so a change in
// them re-renders the pages
const sigs = JSON.parse(execFileSync('python3', ['signatures.py']));
for (const f of readdirSync('../examples').filter((f) => f.endsWith('.py'))) {
	sigs[`examples/${f}`] = readFileSync(`../examples/${f}`, 'utf8').trimEnd();
}

export default defineConfig({
	site: 'https://daenuprobst.github.io',
	base,
	markdown: { remarkPlugins: [[remarkFlodr, { base, sigs }]] },
	integrations: [
		starlight({
			title: 'FloDR',
			description: 'Invertible dimensionality reduction with a normalising flow',
			social: [{ icon: 'github', label: 'GitHub', href: 'https://github.com/daenuprobst/flodr' }],
			editLink: { baseUrl: 'https://github.com/daenuprobst/flodr/edit/main/docs/' },
			customCss: ['./src/styles/custom.css'],
			sidebar: [
				{ label: 'Overview', link: '/' },
				'installation',
				'quickstart',
				{
					label: 'User guide',
					items: [
						'guide/fitting',
						'guide/new-points',
						'guide/inverse',
						'guide/density',
						'guide/audit',
						'guide/visualisation',
						'guide/scikit-learn',
					],
				},
				{ label: 'Examples', items: ['examples/fashion-mnist', 'examples/paul15', 'examples/benchmark'] },
				{ label: 'API', items: ['api/flodr', 'api/viz', 'api/training'] },
			],
		}),
	],
});
