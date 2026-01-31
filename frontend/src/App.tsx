import { useState } from 'react'
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

function App() {
  const [result, setResult] = useState<PredictionResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

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
