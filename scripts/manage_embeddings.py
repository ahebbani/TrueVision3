import argparse
import sys
import numpy as np
from database.db import Database

def main():
    parser = argparse.ArgumentParser(description="Manage face embeddings")
    parser.add_argument("--stats", action="store_true", help="Show embedding stats")
    parser.add_argument("--prune", type=int, metavar="PERSON_ID", help="Prune embeddings for person")
    parser.add_argument("--keep", type=int, default=10, help="Number of templates to keep when pruning")
    parser.add_argument("--delete", type=int, metavar="PERSON_ID", help="Delete all templates for person")
    
    args = parser.parse_args()
    db = Database()
    
    if args.stats:
        faces = db.get_faces()
        for fid, name, _, _ in faces:
            embs = db.get_embeddings(fid)
            if not embs:
                print(f"[{fid}] {name}: 0 templates")
                continue
            quals = [e[2] for e in embs]
            avg_qual = sum(quals) / len(quals)
            print(f"[{fid}] {name}: {len(embs)} templates | Avg Quality: {avg_qual:.1f} | Min: {min(quals):.1f} | Max: {max(quals):.1f}")
            
    elif args.prune:
        fid = args.prune
        embs = db.get_embeddings(fid)
        if len(embs) <= args.keep:
            print(f"Person {fid} has {len(embs)} templates, which is <= keep limit ({args.keep}). Doing nothing.")
            return
            
        embs.sort(key=lambda x: x[2]) # Sort by quality
        num_to_del = len(embs) - args.keep
        to_del_ids = [e[0] for e in embs[:num_to_del]]
        
        db.delete_embeddings(to_del_ids)
        print(f"Deleted {num_to_del} lowest quality templates for person {fid}.")
        
    elif args.delete:
        fid = args.delete
        embs = db.get_embeddings(fid)
        to_del_ids = [e[0] for e in embs]
        db.delete_embeddings(to_del_ids)
        print(f"Deleted all {len(to_del_ids)} templates for person {fid}.")
        
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
