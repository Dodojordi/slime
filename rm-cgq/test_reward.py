"""
Send a test request to the reward server.

Usage:
    python call_reward_api.py --proof            # proof verification
    python call_reward_api.py                    # normal (non-proof) scoring
"""
import argparse
import json
import os

import requests


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="http://100.102.27.140", help="Reward server host (with scheme)")
    parser.add_argument("--port", type=int, default=8001, help="Reward server port")
    parser.add_argument("--proof", action="store_true", help="Use proof verification payload")
    parser.add_argument("--reviews", type=int, default=3, help="Reviews count when using pessimistic proof verifier")
    return parser.parse_args()


def build_payload(use_proof: bool, reviews: int):
    if use_proof:
        return {
            "response": """"Okay, I have analyzed the candidate solutions.
All five candidates arrive at the correct conclusion using the same core argument.
The argument proceeds as follows:
1.  Let $p_k$ be the $k$-th perfect power. $A_n=k$ if and only if $p_k \le n < p_{k+1}$.
2.  We are looking for $n$ such that $A_n=k$ and $k \mid n+2024$. This means $n \in [p_k, p_{k+1}-1]$ and $n \equiv -2024 \pmod k$.
3.  A sufficient condition for the existence of such $n$ for a given $k$ is $p_{k+1}-p_k \ge k$, because if the interval $[p_k, p_{k+1}-1]$ has length at least $k$, it must contain a representative for every residue class modulo $k$.
4.  Let $S=\{k \in \mathbb{Z}^+ \mid p_{k+1}-p_k \ge k\}$. If $S$ is infinite, then there are infinitely many $k$ for which such $n_k$ exists. The $n_k$ values are distinct for distinct $k$.
5.  Prove that $S$ is infinite by contradiction. Assume $S$ is finite. Then $p_{k+1}-p_k \le k-1$ for all $k \ge K$ for some $K$.
6.  This assumption $p_{k+1}-p_k \le k-1$ for $k \ge K$ implies $p_m \le \frac{m^2}{2}+O(m)$ as $m \to \infty$.
7.  However, $A_n \sim \sqrt{n}$ as $n \to \infty$. Since $A_{p_k}=k$, this implies $p_k \sim k^2$, i.e., $\lim_{k \to \infty} p_k/k^2=1$.
8.  The result $p_k \sim k^2$ contradicts $\limsup_{k \to \infty} p_k/k^2 \le 1/2$.
9.  Therefore, $S$ must be infinite, and there are infinitely many $n$ satisfying the condition.

This line of reasoning is sound and correctly implemented in all five candidates.
Candidate 3's assessment of Candidate 1 seems inaccurate, as Candidate 1 does not claim $p_{k+1}-p_k \ge k$ for all $k \ge K_0$.
Candidate 5's assessment of Candidate 2 might be slightly misleading, as Candidate 2 mentions $p_{k+1}-p_k \sim 2k$ as an alternative but uses the contradiction proof.
Despite these minor issues in the assessments provided by Candidates 3 and 5, the solutions themselves appear to be correct.

ALL INPUTS ARE DEFINITELY CORRECT: TRUE

***

**Solution:**

Let $A_n$ denote the number of perfect powers less than or equal to $n$. A perfect power is an integer of the form $a^b$ where $a \ge 1, b \ge 2$ are integers. We want to prove that there are infinitely many positive integers $n$ such that $A_n$ divides $n+2024$.

Let $P=\{p_1, p_2, p_3, \dots\}$ be the set of perfect powers in increasing order. So $p_1=1, p_2=4, p_3=8, p_4=9, \dots$.
By definition, $A_n=k$ if and only if $p_k \le n < p_{k+1}$.
We are looking for $n$ such that $A_n=k$ and $k \mid n+2024$ for some $k$.
For a fixed $k$, $A_n=k$ restricts $n$ to the interval $I_k=[p_k, p_{k+1}-1]$. We need to know if there exists $n \in I_k$ such that $n \equiv -2024 \pmod k$.

The interval $I_k$ contains $p_{k+1}-p_k$ consecutive integers. If the length $p_{k+1}-p_k \ge k$, then $I_k$ contains at least $k$ consecutive integers. In this case, $I_k$ must contain an integer for every residue class modulo $k$. In particular, it must contain an integer $n_k$ such that $n_k \equiv -2024 \pmod k$.
For this $n_k$, we have $A_{n_k}=k$ since $n_k \in I_k$. Thus $A_{n_k} \mid n_k+2024$.

Let $S=\{k \in \mathbb{Z}^+ \mid p_{k+1}-p_k \ge k\}$. If $S$ is an infinite set, then for each $k \in S$, we can find such an $n_k$. The values $n_k$ for $k \in S$ are distinct because if $k_1 < k_2$ are in $S$, then $n_{k_1} \in I_{k_1}$ and $n_{k_2} \in I_{k_2}$. Since $p_{k_1+1} \le p_{k_2}$, we have $n_{k_1} \le p_{k_1+1}-1 < p_{k_2} \le n_{k_2}$, so $n_{k_1} \ne n_{k_2}$.
Thus, if $S$ is infinite, there are infinitely many $n$ satisfying the condition.

We now prove that $S$ is infinite.
Assume for contradiction that $S$ is finite. Then there exists an integer $K$ such that for all $k \ge K$, $p_{k+1}-p_k < k$. This implies $p_{k+1}-p_k \le k-1$ for $k \ge K$.
Let $m > K$. We can write $p_m$ as $p_K+\sum_{j=K}^{m-1}(p_{j+1}-p_j)$.
$p_m \le p_K+\sum_{j=K}^{m-1}(j-1)=p_K+\sum_{l=K-1}^{m-2}l=p_K+\frac{(m-2)(m-1)}{2}-\frac{(K-2)(K-1)}{2}$.
$p_m \le \frac{m^2-3m+2}{2}+C_K$, where $C_K=p_K-\frac{(K-2)(K-1)}{2}$ is a constant.
$p_m \le \frac{1}{2}m^2-\frac{3}{2}m+C_K$.
Dividing by $m^2$, we get $\frac{p_m}{m^2} \le \frac{1}{2}-\frac{3}{2m}+\frac{C_K}{m^2}$.
Taking the limit superior as $m \to \infty$, we get $\limsup_{m \to \infty} \frac{p_m}{m^2} \le \frac{1}{2}$.

Now, let's consider the asymptotic behavior of $A_n$. We know $A_n \ge \lfloor\sqrt{n}\rfloor \ge \sqrt{n}-1$. Also, $A_n \le \sum_{b=2}^{\log_2 n} n^{1/b} \le \sqrt{n}+n^{1/3}\log_2 n$.
Thus, $1-\frac{1}{\sqrt{n}} \le \frac{A_n}{\sqrt{n}} \le 1+\frac{\log_2 n}{n^{1/6}}$. Since both $1/\sqrt{n} \to 0$ and $(\log_2 n)/n^{1/6} \to 0$ as $n \to \infty$, we have $\lim_{n \to \infty} A_n/\sqrt{n}=1$, i.e., $A_n \sim \sqrt{n}$.
Let $n=p_k$. Then $A_{p_k}=k$. So $k \sim \sqrt{p_k}$, which implies $p_k \sim k^2$, i.e., $\lim_{k \to \infty} p_k/k^2=1$.
This contradicts $\limsup_{m \to \infty} p_m/m^2 \le 1/2$.
Therefore, the assumption that $S$ is finite must be false. $S$ must be an infinite set.

Since $S$ is infinite, there are infinitely many $k$ such that $p_{k+1}-p_k \ge k$. For each such $k$, there exists $n_k \in [p_k, p_{k+1}-1]$ such that $n_k \equiv -2024 \pmod k$. For this $n_k$, $A_{n_k}=k$, so $A_{n_k} \mid n_k+2024$. The set $\{n_k \mid k \in S\}$ is infinite.
Thus, there are infinitely many $n$ such that $A_n$ divides $n+2024$.

Final Answer: The final answer is $\boxed{There are infinitely many n such that A_n divides n+2024}$
""",
            "question": "For a positive integer $n$, let $A_{n}$ be the number of perfect powers less than or equal to $n$. Here, a perfect power is a number that can be expressed in the form $a^{b}$, where $a$ is a positive integer and $b$ is an integer greater than or equal to 2. Prove that there are infinitely many $n$ such that $A_{n}$ divides $n+2024$.",
            "is_proof": True,
            "reviewer": "progressive",
            "reviews": reviews,
            "label": None,
            "points": None
        }
    return {
        # "response": "The answer is 42.",
        "response": r"The answer is <answer>\boxed{42}</answer>",
        "label": ["42"],
        "question": "What is 6 * 7?",
        "use_xverify": True,
    }


def main():
    args = parse_args()
    url = f"{args.host}:{args.port}/"
    payload = build_payload(args.proof, args.reviews)

    print(f"POST {url}")
    print("Payload:")
    print(json.dumps(payload, ensure_ascii=False, indent=2))

    resp = requests.post(url, json=payload, timeout=60)
    print(f"\nStatus: {resp.status_code}")
    try:
        print("Response JSON:")
        print(json.dumps(resp.json(), ensure_ascii=False, indent=2))
    except Exception:
        print("Raw response:")
        print(resp.text)


if __name__ == "__main__":
    main()
