// @ts-check
import {themes as prismThemes} from 'prism-react-renderer';

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'FloDR',
  tagline: 'Invertible dimensionality reduction with a normalising flow',
  url: 'https://daenuprobst.github.io',
  baseUrl: '/flodr/',
  organizationName: 'daenuprobst',
  projectName: 'flodr',
  trailingSlash: false,
  onBrokenLinks: 'throw',
  markdown: {hooks: {onBrokenMarkdownLinks: 'throw'}},
  i18n: {defaultLocale: 'en', locales: ['en']},
  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          routeBasePath: '/',
          sidebarPath: './sidebars.js',
          editUrl: 'https://github.com/daenuprobst/flodr/edit/main/docs/',
        },
        blog: false,
        theme: {customCss: './src/css/custom.css'},
      }),
    ],
  ],
  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      navbar: {
        title: 'FloDR',
        items: [
          {type: 'docSidebar', sidebarId: 'guide', position: 'left', label: 'Guide'},
          {type: 'docSidebar', sidebarId: 'api', position: 'left', label: 'API'},
          {href: 'https://github.com/daenuprobst/flodr', label: 'GitHub', position: 'right'},
        ],
      },
      footer: {
        style: 'light',
        copyright: `FloDR is released under the MIT licence.`,
      },
      prism: {
        theme: prismThemes.github,
        darkTheme: prismThemes.dracula,
        additionalLanguages: ['bash'],
      },
    }),
};

export default config;
