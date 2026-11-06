import sf from 'slopfab';

export default ([model, fileId, destination, ...extra]) => {
  const catalog = sf.fs.readJson(sf.path.join(sf.scriptDirectory, 'catalog.json'));
  const weightsDirectory = sf.path.join(sf.path.dirname(sf.executable), 'weights');
  if (!model) return {models: Object.keys(catalog), weightsDirectory};
  const entry = catalog[model];
  if (!entry || typeof entry !== 'object' || !Object.hasOwn(catalog, model))
    throw Error(`Unknown model: ${model}. Available: ${Object.keys(catalog).join(', ')}`);
  if (!fileId && !destination && extra.length === 0) return {model, ...entry, weightsDirectory};
  if (!fileId || extra.length) throw Error('Usage: slopfab download MODEL [FILE_ID [DESTINATION]]');
  const file = entry.files.find(item => item.id === fileId);
  if (!file) throw Error(`Unknown file: ${fileId}. List files with slopfab download ${model}`);
  if (destination === undefined) {
    const filename = file.filename ?? decodeURIComponent(file.url.split(/[?#]/)[0].split('/').pop());
    if (typeof filename !== 'string' || !filename || filename === '.' || filename === '..' || /[\\/:]/.test(filename))
      throw Error(`Invalid download filename for ${model}/${fileId}`);
    destination = sf.path.join(weightsDirectory, filename);
  }
  sf.fs.mkdir(sf.path.dirname(sf.path.resolve(destination)));
  const downloaded = sf.http.download(file.url, destination,
    {timeoutMs: 7200000, resume: true, connections: 4, retries: 2});
  return {model, file: file.id, path: downloaded.path, bytes: downloaded.bytes,
    connections: downloaded.connections, resumedBytes: downloaded.resumedBytes,
    sha256: sf.sha256(downloaded.path)};
};
