import asyncio
import json
import pickle
import numpy as np
from sqlalchemy import select
from polyflip.db.connection import get_db_session
from polyflip.db.models import ModelRegistry

async def main():
    async for session in get_db_session():
        # Get latest active model for XRP and ETH
        assets = ['XRP', 'ETH']
        for asset in assets:
            stmt = select(ModelRegistry).where(
                ModelRegistry.asset == asset,
                ModelRegistry.is_active == True
            ).order_by(ModelRegistry.version.desc()).limit(1)
            result = await session.execute(stmt)
            model_record = result.scalar_one_or_none()
            
            if model_record:
                print(f"\n=== {asset} v{model_record.version} ===")
                features = model_record.features.split(',')
                print(f"Number of features: {len(features)}")
                
                pipeline = pickle.loads(model_record.model_blob)
                
                logreg = None
                # Check if it's CalibratedClassifierCV
                if type(pipeline).__name__ == 'CalibratedClassifierCV':
                    base = pipeline.estimator
                    if hasattr(base, 'named_steps'):
                        logreg = base.named_steps['classifier']
                    else:
                        logreg = base
                elif hasattr(pipeline, 'named_steps'):
                    logreg = pipeline.named_steps['classifier']
                    
                if logreg:
                    try:
                        weights = logreg.coef_[0]
                        fi = sorted(zip(features, weights), key=lambda x: abs(x[1]), reverse=True)
                        print("Weights:")
                        for f, w in fi:
                            print(f"{f}: {w:.6f}")
                    except Exception as e:
                        print("Could not extract weights directly:", e)
        break

if __name__ == "__main__":
    asyncio.run(main())
