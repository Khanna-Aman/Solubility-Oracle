import { useState, FormEvent } from 'react'
import './PredictionForm.css'

interface PredictionFormProps {
  onSubmit: (smiles: string) => void
  loading: boolean
}

function PredictionForm({ onSubmit, loading }: PredictionFormProps) {
  const [smiles, setSmiles] = useState('')

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault()
    if (smiles.trim()) {
      onSubmit(smiles.trim())
    }
  }

  return (
    <form onSubmit={handleSubmit} className="prediction-form">
      <div className="form-group">
        <label htmlFor="smiles">SMILES String</label>
        <input
          type="text"
          id="smiles"
          value={smiles}
          onChange={(e) => setSmiles(e.target.value)}
          placeholder="e.g., CC(=O)OC1=CC=CC=C1C(=O)O (Aspirin)"
          className="smiles-input"
          required
          disabled={loading}
        />
        <small className="help-text">
          Enter a valid SMILES string representing the molecular structure
        </small>
      </div>
      
      <button 
        type="submit" 
        className="predict-button"
        disabled={loading || !smiles.trim()}
      >
        {loading ? 'Predicting...' : 'Predict Solubility'}
      </button>
    </form>
  )
}

export default PredictionForm
