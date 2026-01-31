import { PredictionResult } from '../App'
import './Results.css'

interface ResultsProps {
  result: PredictionResult
}

function Results({ result }: ResultsProps) {
  const getSolubilityInterpretation = (logs: number): string => {
    if (logs > 0) return 'Highly Soluble'
    if (logs > -2) return 'Moderately Soluble'
    if (logs > -4) return 'Poorly Soluble'
    return 'Very Poorly Soluble'
  }

  const getSolubilityColor = (logs: number): string => {
    if (logs > 0) return '#48bb78'
    if (logs > -2) return '#38b2ac'
    if (logs > -4) return '#ed8936'
    return '#f56565'
  }

  return (
    <div className="results">
      <h2 className="results-title">Prediction Results</h2>
      
      <div className="result-card">
        <div className="result-item">
          <span className="result-label">SMILES:</span>
          <code className="result-value-smiles">{result.smiles}</code>
        </div>
        
        <div className="result-item">
          <span className="result-label">Predicted LogS:</span>
          <span 
            className="result-value-logs"
            style={{ color: getSolubilityColor(result.predicted_logs) }}
          >
            {result.predicted_logs.toFixed(4)}
          </span>
        </div>
        
        <div className="result-item">
          <span className="result-label">Solubility:</span>
          <span 
            className="result-value-interpretation"
            style={{ color: getSolubilityColor(result.predicted_logs) }}
          >
            {getSolubilityInterpretation(result.predicted_logs)}
          </span>
        </div>
      </div>
      
      {result.message && (
        <div className="result-message">
          {result.message}
        </div>
      )}
    </div>
  )
}

export default Results
