import { useState } from 'react'
import { uploadPdf } from './services/api.js'

export default function App() {
  const [file, setFile] = useState(null)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  async function handleUpload(event) {
    event.preventDefault()
    if (!file) {
      setError('Choose a PDF before uploading.')
      return
    }
    setLoading(true)
    setError('')
    setResult(null)
    try {
      setResult(await uploadPdf(file))
    } catch (err) {
      setError(err.message || 'Unable to reach the API. Check that the backend is running.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <main className="page-shell">
      <nav className="topbar"><a className="brand" href="#top"><span className="brand-mark">D</span> DocuQuery</a><span className="topbar-note">DOCUMENT INTELLIGENCE</span></nav>
      <section className="hero" id="top">
        <div className="eyebrow"><span className="status-dot" /> YOUR DOCUMENTS, UNDERSTOOD</div>
        <h1>Find the signal<br /><span>inside your documents.</span></h1>
        <p className="hero-copy">Upload a PDF to extract page-aware text and prepare it for evidence-grounded question answering.</p>
      </section>
      <section className="workspace">
        <div className="section-heading"><div><p className="overline">01 / INGESTION</p><h2>Add a document</h2></div><span className="step-tag">PDF · MAX 15 MB</span></div>
        <form className="upload-form" onSubmit={handleUpload}>
          <label className="dropzone">
            <input type="file" accept="application/pdf,.pdf" onChange={e => { setFile(e.target.files?.[0] || null); setError(''); setResult(null) }} />
            <span className="upload-icon">↑</span>
            <strong>{file ? file.name : 'Choose a PDF to get started'}</strong>
            <span className="dropzone-help">{file ? `${(file.size / 1024 / 1024).toFixed(2)} MB · Ready to upload` : 'Select a file from your device · PDF only'}</span>
          </label>
          <button className="primary-button" type="submit" disabled={loading || !file}>{loading ? 'Extracting text…' : 'Upload and extract'} <span>↗</span></button>
        </form>
        {error && <div className="message error" role="alert">{error}</div>}
        {result && <section className="result-panel" aria-live="polite">
          <div className="success-heading"><span className="success-check">✓</span><div><strong>Document processed</strong><p>Your PDF text is ready for the next pipeline stage.</p></div></div>
          <div className="stats"><div><span>Pages</span><strong>{result.page_count}</strong></div><div><span>Chunks</span><strong>{result.chunk_count}</strong></div><div><span>Source</span><strong className="source-value">{result.filename}</strong></div></div>
          <div className="chunk-list"><h3>Extracted passages</h3>{result.chunks.slice(0, 8).map(chunk => <article className="chunk" key={chunk.chunk_id}><div className="chunk-meta"><span>PAGE {chunk.page_number}</span><code>{chunk.chunk_id}</code></div><p>{chunk.text}</p></article>)}{result.chunks.length > 8 && <p className="more-note">Showing 8 of {result.chunks.length} chunks.</p>}</div>
        </section>}
      </section>
      <footer><span>DOCUQUERY / MVP</span><span>Evidence first. Answers you can inspect.</span></footer>
    </main>
  )
}
