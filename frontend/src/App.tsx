import { useState, useEffect } from 'react'
import './App.css'
import PredictionForm from './components/PredictionForm'
import Header from './components/Header'
import Results from './components/Results'

export interface PredictionResult {
  smiles: string
  predicted_logs: number
  success: boolean
  message: string
}

interface ModelStatus {
  model_loaded: boolean
  model_type: string
  training_complete: boolean
}

function App() {
  const [result, setResult] = useState<PredictionResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [modelStatus, setModelStatus] = useState<ModelStatus | null>(null)

  useEffect(() => {
    // Check model status on component mount
    const checkModelStatus = async () => {
      try {
        const response = await fetch('http://localhost:8050/api/v1/health')
        if (response.ok) {
          const data = await response.json()
          setModelStatus({
            model_loaded: data.model_loaded || false,
            model_type: data.model_type || 'AttentiveFP Ensemble',
            training_complete: data.model_loaded || false
          })
        }
      } catch (err) {
        console.log('Could not fetch model status')
      }
    }
    checkModelStatus()
  }, [])

  const handlePrediction = async (smiles: string) => {
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const response = await fetch('http://localhost:8050/api/v1/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ smiles }),
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || 'Prediction failed')
      }

      const data = await response.json()
      setResult(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app">
      <Header />
      <main className="main-content">
        <div className="container">
          <h1 className="title">💧 Solubility Oracle</h1>
          <p className="subtitle">
            AI-Powered Aqueous Solubility Prediction using AttentiveFP
          </p>

          {modelStatus && (
            <div className="model-status">
              <div className="status-item">
                <span className="status-label">Model Status:</span>
                <span className={`status-badge ${modelStatus.model_loaded ? 'status-ready' : 'status-loading'}`}>
                  {modelStatus.model_loaded ? '✓ Ready' : '⏳ Loading...'}
                </span>
              </div>
              <div className="status-item">
                <span className="status-label">Architecture:</span>
                <span className="status-value">{modelStatus.model_type}</span>
              </div>
              {modelStatus.training_complete && (
                <div className="status-item">
                  <span className="status-label">Training:</span>
                  <span className="status-badge status-complete">✓ Complete</span>
                </div>
              )}
            </div>
          )}

          <PredictionForm
            onSubmit={handlePrediction}
            loading={loading}
          />

          {error && (
            <div className="error-message">
              <strong>Error:</strong> {error}
            </div>
          )}

          {result && <Results result={result} />}
        </div>
      </main>
    </div>
  )
}

export default App
