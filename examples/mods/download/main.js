import sf from 'slopfab';

export default ([model, fileId, destination, ...extra]) => {
  const catalog = sf.fs.readJson(sf.path.join(sf.scriptDirectory, 'catalog.json'));
  if (!model) return {models: Object.keys(catalog)};
  const entry = catalog[model];
  if (!entry || typeof entry !== 'object' || !Object.hasOwn(catalog, model))
    throw Error(`Unknown model: ${model}. Available: ${Object.keys(catalog).join(', ')}`);
  if (!fileId && !destination && extra.length === 0) return {model, ...entry};
  if (!fileId || !destination || extra.length) throw Error('Usage: slopfab download MODEL [FILE_ID DESTINATION]');
  const file = entry.files.find(item => item.id === fileId);
  if (!file) throw Error(`Unknown file: ${fileId}. List files with slopfab download ${model}`);
  sf.fs.mkdir(sf.path.dirname(sf.path.resolve(destination)));
  const downloaded = sf.http.download(file.url, destination,
    {timeoutMs: 7200000, resume: true, connections: 4, retries: 2});
  return {model, file: file.id, path: downloaded.path, bytes: downloaded.bytes,
    connections: downloaded.connections, resumedBytes: downloaded.resumedBytes,
    sha256: sf.sha256(downloaded.path)};
};
