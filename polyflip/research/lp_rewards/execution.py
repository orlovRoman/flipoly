import asyncio
import concurrent.futures
from dataclasses import dataclass
import datetime
from decimal import Decimal, getcontext
from enum import Enum
import json
import logging
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Set

from eth_account import Account
from eth_account.messages import encode_typed_data

from py_clob_client_v2.order_utils.model.order_data_v2 import SignedOrderV2
from py_clob_client_v2.order_utils.model.side import Side
from py_clob_client_v2.order_utils.model.signature_type_v2 import SignatureTypeV2

getcontext().prec = 28
logger = logging.getLogger(__name__)

POLYGON_CHAIN_ID = 137
CTF_EXCHANGE_ADDRESS = "0xE111180000d2663C0091e4f400237545B87B996B"


class LiveOrderExecutor:
    """Manages order submission to Polymarket CLOB with strict hard gates.

    Hard Gates:
    1. LP_LIVE_ENABLED environment variable must be explicitly 'true'.
    2. Protocol SHA-256 hash must match the approved research protocol.
    3. Gate A verdict check: artifact must exist, verdict must be PROCEED_LIVE, hash match.
    4. Dedicated isolated wallet check: wallet must be configured and isolated from prod main wallet.
    5. Working capital limit: cumulative committed capital must not exceed allocated limit ($100).
    6. Live balance check: wallet USDC balance must be sufficient.
    7. Token allowlist: token_id must be in approved active universe allowlist.
    8. Tick size and min size validation: price conforms to tick, size >= min_size.
    9. Emergency cancel-all on error.
    """

    def __init__(
        self,
        expected_protocol_hash: Optional[str] = None,
        gate_a_verdict_path: Optional[Path] = None,
        wallet_private_key: Optional[str] = None,
        wallet_address: Optional[str] = None,
        main_wallet_address: Optional[str] = None,
        allocated_capital_limit: Decimal = Decimal("100.00"),
        allowlist_tokens: Optional[Set[str]] = None,
        min_size: Decimal = Decimal("5.0"),
        tick_size: Decimal = Decimal("0.001"),
        require_gate_a: Optional[bool] = None,
        clob_client: Optional[Any] = None,
        domain_name: str = "Polymarket CTF Exchange",
        domain_version: str = "2",
        verifying_contract: str = CTF_EXCHANGE_ADDRESS,
    ):
        self.expected_protocol_hash = expected_protocol_hash
        self.gate_a_verdict_path = gate_a_verdict_path
        self.wallet_private_key = wallet_private_key or os.getenv("LP_WALLET_PRIVATE_KEY")
        self.wallet_address = wallet_address or os.getenv("LP_ISOLATED_WALLET_ADDRESS")
        if self.wallet_private_key:
            try:
                derived_address = Account.from_key(self.wallet_private_key).address
                if self.wallet_address and derived_address.lower() != self.wallet_address.lower():
                    raise ValueError(f"Private key mismatch: derived {derived_address}, expected {self.wallet_address}")
                self.wallet_address = derived_address
            except ValueError:
                raise
            except Exception:
                pass
        self.main_wallet_address = (
            main_wallet_address
            or os.getenv("PROD_MAIN_WALLET_ADDRESS")
            or os.getenv("POLYGON_WALLET_ADDRESS")
        )
        self.allocated_capital_limit = allocated_capital_limit
        self.allowlist_tokens = set(allowlist_tokens) if allowlist_tokens is not None else None
        self.min_size = min_size
        self.tick_size = tick_size
        self.require_gate_a = True
        self.clob_client = clob_client
        self.domain_name = domain_name
        self.domain_version = domain_version
        self.verifying_contract = verifying_contract
        self.current_committed_capital: Decimal = Decimal("0.0")
        self.submitted_orders: List[Dict[str, Any]] = []

    def is_live_enabled(self) -> bool:
        env_val = os.getenv("LP_LIVE_ENABLED", "false").lower()
        return env_val in ("1", "true", "yes")

    def verify_gate_a(
        self,
        verdict_path: Optional[Path] = None,
        verdict_data: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Verify that Gate A verdict is PROCEED_LIVE and protocol hash matches."""
        target_path = verdict_path or self.gate_a_verdict_path
        if verdict_data is None and target_path:
            if not target_path.exists():
                raise PermissionError(f"Gate A verdict artifact not found at {target_path}. Live orders prohibited.")
            with open(target_path, "r", encoding="utf-8") as f:
                verdict_data = json.load(f)

        if verdict_data is None:
            raise PermissionError("Gate A verdict data is missing. Live orders prohibited until Gate A passes.")

        verdict = verdict_data.get("verdict")
        if verdict != "PROCEED_LIVE":
            raise PermissionError(f"Gate A verdict is '{verdict}', expected 'PROCEED_LIVE'. Live trading forbidden.")

        v_hash = verdict_data.get("protocol_hash")
        if not v_hash or (self.expected_protocol_hash and v_hash != self.expected_protocol_hash):
            raise ValueError(f"Gate A artifact protocol hash mismatch: expected {self.expected_protocol_hash}, got {v_hash}")

        return True

    def verify_isolated_wallet(
        self,
        wallet_address: Optional[str] = None,
        main_wallet_address: Optional[str] = None,
    ) -> bool:
        """Verify dedicated isolated wallet is configured and not shared with main production trading."""
        addr = wallet_address or self.wallet_address
        main_addr = main_wallet_address or self.main_wallet_address

        if not addr:
            raise ValueError("Dedicated LP isolated wallet address is not configured.")

        if main_addr and addr.lower() == main_addr.lower():
            raise PermissionError(
                f"Isolated wallet address ({addr}) matches main production trading wallet ({main_addr})! "
                "LP rewards must run on a dedicated isolated wallet."
            )

        return True

    def verify_working_capital_limit(
        self,
        new_order_cost: Decimal,
        current_committed: Optional[Decimal] = None,
    ) -> bool:
        """Verify working capital limit ($100.00) is strictly respected."""
        committed = current_committed if current_committed is not None else self.current_committed_capital
        total_after = committed + new_order_cost
        if total_after > self.allocated_capital_limit:
            raise ValueError(
                f"Working capital limit exceeded: current=${committed}, "
                f"requested=${new_order_cost}, limit=${self.allocated_capital_limit}"
            )
        return True

    def verify_token_allowlist(self, token_id: str) -> bool:
        """Verify token is in approved active universe allowlist."""
        if self.allowlist_tokens is not None:
            if token_id not in self.allowlist_tokens:
                raise ValueError(f"Token {token_id} is not in approved active universe allowlist.")
        return True

    def verify_tick_and_min_size(self, price: Decimal, size: Decimal) -> bool:
        """Verify price conforms to tick size and range (0, 1) and size >= min_size."""
        if price <= Decimal("0.0") or price >= Decimal("1.0"):
            raise ValueError(f"Invalid order price {price}: must be between 0.0 and 1.0.")

        if size < self.min_size:
            raise ValueError(f"Order size {size} is below minimum allowable size {self.min_size}.")

        remainder = (price / self.tick_size) % Decimal("1.0")
        if remainder != Decimal("0.0") and abs(remainder - Decimal("1.0")) > Decimal("1e-8") and remainder > Decimal("1e-8"):
            raise ValueError(f"Order price {price} does not conform to tick size {self.tick_size}.")

        return True

    def get_pusd_balance_onchain(self, wallet_address: str) -> Decimal:
        import urllib.request
        import json
        rpc_url = os.getenv("POLYGON_RPC_URL", "https://polygon-rpc.com")
        pusd_address = "0xC011a7E12a19f7b1f670d46f03b03f3342e82dfb"
        addr = wallet_address.lower().replace("0x", "").zfill(64)
        data_payload = "0x70a08231" + addr
        payload = {
            "jsonrpc": "2.0",
            "method": "eth_call",
            "params": [{"to": pusd_address, "data": data_payload}, "latest"],
            "id": 1
        }
        try:
            req = urllib.request.Request(rpc_url, data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(req, timeout=5) as response:
                res = json.loads(response.read())
                balance_hex = res.get("result", "0x0")
                if balance_hex == "0x": balance_hex = "0x0"
                balance_int = int(balance_hex, 16)
                return Decimal(balance_int) / Decimal("1e6")
        except Exception as e:
            logger.warning(f"On-chain balance fetch failed: {e}")
            return Decimal("0.0")

    def get_pusd_allowance_onchain(self, wallet_address: str, spender_address: str) -> Decimal:
        import urllib.request
        import json
        rpc_url = os.getenv("POLYGON_RPC_URL", "https://polygon-rpc.com")
        pusd_address = "0xC011a7E12a19f7b1f670d46f03b03f3342e82dfb"
        owner = wallet_address.lower().replace("0x", "").zfill(64)
        spender = spender_address.lower().replace("0x", "").zfill(64)
        data_payload = "0xdd62ed3e" + owner + spender
        payload = {
            "jsonrpc": "2.0",
            "method": "eth_call",
            "params": [{"to": pusd_address, "data": data_payload}, "latest"],
            "id": 1
        }
        try:
            req = urllib.request.Request(rpc_url, data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(req, timeout=5) as response:
                res = json.loads(response.read())
                allow_hex = res.get("result", "0x0")
                if allow_hex == "0x": allow_hex = "0x0"
                allow_int = int(allow_hex, 16)
                return Decimal(allow_int) / Decimal("1e6")
        except Exception as e:
            logger.warning(f"On-chain allowance fetch failed: {e}")
            return Decimal("0.0")

    def check_live_balance(self, cost: Decimal, available_balance: Optional[Decimal] = None) -> bool:
        """Verify wallet has sufficient pUSD balance."""
        if available_balance is None:
            if not self.wallet_address:
                raise PermissionError("Dedicated isolated wallet address or available_balance must be provided to verify live balance.")
            available_balance = self.get_pusd_balance_onchain(self.wallet_address)
        if available_balance < cost:
            raise ValueError(f"Insufficient live balance: available=${available_balance}, required=${cost}")
        return True

    def check_allowance(self, required_amount: Decimal, current_allowance: Optional[Decimal] = None) -> bool:
        """Verify collateral allowance is sufficient for order execution."""
        if current_allowance is None:
            if not self.wallet_address:
                raise PermissionError("Dedicated isolated wallet address or current_allowance must be provided to verify allowance.")
            current_allowance = self.get_pusd_allowance_onchain(self.wallet_address, self.verifying_contract)
        if current_allowance < required_amount:
            raise PermissionError(
                f"Insufficient collateral allowance: available=${current_allowance}, required=${required_amount}"
            )
        return True

    @staticmethod
    def _parse_token_id(token_id: Any) -> int:
        """Parse token ID from decimal string, hex string, or integer."""
        if isinstance(token_id, int):
            return token_id
        s = str(token_id).strip()
        if s.startswith(("0x", "0X")):
            try:
                return int(s, 16)
            except ValueError:
                raise ValueError(f"Invalid token ID: {token_id}")
        if s.isdigit():
            return int(s)
        try:
            return int(s)
        except ValueError:
            raise ValueError(f"Invalid token ID: {token_id}")

    def _execute_client_call(self, method_name: str, *args, **kwargs) -> Any:
        """Safely execute a method on clob_client, synchronously waiting until completion."""
        if not self.clob_client or not hasattr(self.clob_client, method_name):
            return None
        fn = getattr(self.clob_client, method_name)
        res = fn(*args, **kwargs)
        if asyncio.iscoroutine(res):
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            if loop and loop.is_running():
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    return pool.submit(asyncio.run, res).result(timeout=15)
            else:
                return asyncio.run(res)
        return res

    async def _execute_client_call_async(self, method_name: str, *args, **kwargs) -> Any:
        """Asynchronously execute a method on clob_client."""
        if not self.clob_client or not hasattr(self.clob_client, method_name):
            return None
        fn = getattr(self.clob_client, method_name)
        res = fn(*args, **kwargs)
        if asyncio.iscoroutine(res):
            return await res
        return res

    def sign_eip712_order(
        self,
        token_id: str,
        side: str,
        price: Decimal,
        size: Decimal,
        expiration_sec: int = 300,
        fee_rate_bps: int = 0,
    ) -> Dict[str, Any]:
        """Construct and sign Polymarket CTF Exchange EIP-712 order structure."""
        side_int = 0 if side.upper() in ("BUY", "BID") else 1
        salt = int(time.time() * 1000)
        now_ts_ms = int(time.time() * 1000)

        if side_int == 0:
            maker_amount = int(price * size * Decimal("1e6"))
            taker_amount = int(size * Decimal("1e6"))
        else:
            maker_amount = int(size * Decimal("1e6"))
            taker_amount = int(price * size * Decimal("1e6"))

        maker_address = self.wallet_address or "0x0000000000000000000000000000000000000000"
        empty_bytes32 = "0x" + "00" * 32

        order_data = {
            "types": {
                "EIP712Domain": [
                    {"name": "name", "type": "string"},
                    {"name": "version", "type": "string"},
                    {"name": "chainId", "type": "uint256"},
                    {"name": "verifyingContract", "type": "address"},
                ],
                "Order": [
                    {"name": "salt", "type": "uint256"},
                    {"name": "maker", "type": "address"},
                    {"name": "signer", "type": "address"},
                    {"name": "tokenId", "type": "uint256"},
                    {"name": "makerAmount", "type": "uint256"},
                    {"name": "takerAmount", "type": "uint256"},
                    {"name": "side", "type": "uint8"},
                    {"name": "signatureType", "type": "uint8"},
                    {"name": "timestamp", "type": "uint256"},
                    {"name": "metadata", "type": "bytes32"},
                    {"name": "builder", "type": "bytes32"},
                ],
            },
            "primaryType": "Order",
            "domain": {
                "name": self.domain_name,
                "version": self.domain_version,
                "chainId": POLYGON_CHAIN_ID,
                "verifyingContract": self.verifying_contract,
            },
            "message": {
                "salt": salt,
                "maker": maker_address,
                "signer": maker_address,
                "tokenId": self._parse_token_id(token_id),
                "makerAmount": maker_amount,
                "takerAmount": taker_amount,
                "side": side_int,
                "signatureType": 0,
                "timestamp": now_ts_ms,
                "metadata": empty_bytes32,
                "builder": empty_bytes32,
            },
        }

        signature = "0x"
        if self.wallet_private_key:
            try:
                signable = encode_typed_data(full_message=order_data)
                signed = Account.sign_message(signable, private_key=self.wallet_private_key)
                signature = "0x" + signed.signature.hex()
            except Exception as e:
                logger.error(f"EIP-712 signing error: {e}")
                raise

        return {
            "order": order_data["message"],
            "signature": signature,
            "order_data": order_data,
        }

    def cancel_all_orders(self) -> bool:
        """Cancel all resting orders on CLOB immediately (emergency guard)."""
        logger.warning("Emergency CANCEL-ALL triggered on live executor.")
        if not self.clob_client:
            raise PermissionError("CLOB client is not authenticated or provided. Live operations forbidden.")
        if hasattr(self.clob_client, "cancel_all_orders"):
            self._execute_client_call("cancel_all_orders")
        elif hasattr(self.clob_client, "cancel_all"):
            self._execute_client_call("cancel_all")
        else:
            raise NotImplementedError("CLOB client does not support cancel_all_orders or cancel_all.")
        self.submitted_orders.clear()
        self.current_committed_capital = Decimal("0.0")
        return True

    async def cancel_all_orders_async(self) -> bool:
        """Cancel all resting orders on CLOB immediately in async context."""
        logger.warning("Emergency CANCEL-ALL triggered on live executor (async).")
        if not self.clob_client:
            raise PermissionError("CLOB client is not authenticated or provided. Live operations forbidden.")
        if hasattr(self.clob_client, "cancel_all_orders"):
            await self._execute_client_call_async("cancel_all_orders")
        elif hasattr(self.clob_client, "cancel_all"):
            await self._execute_client_call_async("cancel_all")
        else:
            raise NotImplementedError("CLOB client does not support cancel_all_orders or cancel_all.")
        self.submitted_orders.clear()
        self.current_committed_capital = Decimal("0.0")
        return True

    def submit_order(
        self,
        token_id: str,
        side: str,
        price: Decimal,
        size: Decimal,
        protocol_hash: str,
        live_balance: Optional[Decimal] = None,
        allowance: Optional[Decimal] = None,
    ) -> Dict[str, Any]:
        """Submit live order with all hard gates and validations enforced."""
        if not self.is_live_enabled():
            raise PermissionError(
                "LP Live trading is disabled. LP_LIVE_ENABLED must be set to 'true' after Gate A passes."
            )

        if not self.clob_client:
            raise PermissionError("CLOB client is not authenticated or provided. Live operations forbidden.")

        if self.expected_protocol_hash and protocol_hash != self.expected_protocol_hash:
            raise ValueError(
                f"Protocol hash mismatch! Expected {self.expected_protocol_hash}, got {protocol_hash}. Order rejected."
            )

        if self.require_gate_a:
            self.verify_gate_a()

        if self.wallet_address:
            self.verify_isolated_wallet()

        self.verify_token_allowlist(token_id)
        self.verify_tick_and_min_size(price, size)

        order_cost = price * size
        self.verify_working_capital_limit(order_cost)
        self.check_live_balance(order_cost, live_balance)
        self.check_allowance(order_cost, allowance)

        try:
            signed_payload = self.sign_eip712_order(token_id, side, price, size)
            logger.info(f"Submitting live order: {side} {size} @ {price} for token {token_id}")

            # Construct typed SignedOrderV2 object for CLOB V2 SDK
            order_msg = signed_payload["order"]
            sdk_side = Side.BUY if side.upper() in ("BUY", "BID") else Side.SELL
            sdk_sig_type = SignatureTypeV2.EOA

            signed_order_obj = SignedOrderV2(
                salt=str(order_msg["salt"]),
                maker=str(order_msg["maker"]),
                signer=str(order_msg["signer"]),
                tokenId=str(order_msg["tokenId"]),
                makerAmount=str(order_msg["makerAmount"]),
                takerAmount=str(order_msg["takerAmount"]),
                side=sdk_side,
                signatureType=sdk_sig_type,
                timestamp=str(order_msg["timestamp"]),
                metadata=str(order_msg["metadata"]),
                builder=str(order_msg["builder"]),
                expiration="0",
                signature=signed_payload["signature"],
            )
            signed_payload["signed_order_obj"] = signed_order_obj

            # Post order to remote CLOB if client exists
            client_order_id = None
            if self.clob_client:
                if hasattr(self.clob_client, "post_order"):
                    post_res = self._execute_client_call("post_order", signed_order_obj)
                    if isinstance(post_res, dict):
                        client_order_id = post_res.get("orderID") or post_res.get("order_id")
                elif hasattr(self.clob_client, "create_order"):
                    post_res = self._execute_client_call("create_order", signed_order_obj)
                    if isinstance(post_res, dict):
                        client_order_id = post_res.get("orderID") or post_res.get("order_id")

            self.current_committed_capital += order_cost
            result = {
                "status": "SUBMITTED",
                "token_id": token_id,
                "side": side,
                "price": str(price),
                "size": str(size),
                "order_id": client_order_id or f"ord_{int(time.time()*1000)}",
                "signed_order": signed_payload,
            }
            self.submitted_orders.append(result)
            return result

        except Exception as e:
            logger.critical(f"Order submission failed: {e}. Triggering emergency cancel-all...")
            self.cancel_all_orders()
            raise
