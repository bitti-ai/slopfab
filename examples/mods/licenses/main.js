import sf from 'slopfab';

export default (args) => {
  if (args.length) throw Error('Usage: slopfab licenses');
  const licenses = sf.fs.readJson(sf.path.join(sf.scriptDirectory, 'licenses.json'));
  for (const {source, text} of licenses) sf.log(`=== ${source} ===\n\n${text}`);
  return {licenses: licenses.map(({source}) => source)};
};
