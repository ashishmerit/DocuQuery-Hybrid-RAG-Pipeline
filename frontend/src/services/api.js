const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

export async function uploadPdf(file) {
  const formData = new FormData()
  formData.append('file', file)
  const response = await fetch(`${API_BASE_URL}/api/documents`, {
    method: 'POST',
    body: formData,
  })
  const payload = await response.json().catch(() => ({}))
  if (!response.ok) {
    throw new Error(payload.detail || `Upload failed (${response.status})`)
  }
  return payload
}
