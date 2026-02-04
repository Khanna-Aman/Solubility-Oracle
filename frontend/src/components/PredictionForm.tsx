import { useState, FormEvent } from 'react'
import './PredictionForm.css'

interface PredictionFormProps {
  onSubmit: (smiles: string) => void
  loading: boolean
}

const EXAMPLE_MOLECULES = [
  { name: 'Aspirin', smiles: 'CC(=O)OC1=CC=CC=C1C(=O)O' },
  { name: 'Caffeine', smiles: 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C' },
  { name: 'Ibuprofen', smiles: 'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O' },
  { name: 'Ethanol', smiles: 'CCO' },
  { name: 'Benzene', smiles: 'c1ccccc1' },
  { name: 'Glucose', smiles: 'C(C1C(C(C(C(O1)O)O)O)O)O' },
]

function PredictionForm({ onSubmit, loading }: PredictionFormProps) {
  const [smiles, setSmiles] = useState('')

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault()
    if (smiles.trim()) {
      onSubmit(smiles.trim())
    }
  }

  const handleExampleClick = (exampleSmiles: string) => {
    setSmiles(exampleSmiles)
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

      <div className="examples-section">
        <label>Example Molecules:</label>
        <div className="examples-grid">
          {EXAMPLE_MOLECULES.map((molecule) => (
            <button
              key={molecule.name}
              type="button"
              className="example-button"
              onClick={() => handleExampleClick(molecule.smiles)}
              disabled={loading}
            >
              {molecule.name}
            </button>
          ))}
        </div>
      </div>

      <button
        type="submit"
        className="predict-button"
        disabled={loading || !smiles.trim()}
      >
        {loading ? '🔄 Predicting...' : '🔮 Predict Solubility'}
      </button>
    </form>
  )
}

export default PredictionForm
