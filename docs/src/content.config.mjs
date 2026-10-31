import { docsSchema } from '@astrojs/starlight/schema';
import { defineCollection } from 'astro:content';
import { glob } from 'astro/loaders';

// the pages are the markdown files in docs/, read where they are
export const collections = {
	docs: defineCollection({ loader: glob({ base: './docs', pattern: '**/*.md' }), schema: docsSchema() }),
};
