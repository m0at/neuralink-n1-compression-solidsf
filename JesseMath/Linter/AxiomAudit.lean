import Lean

open Lean Elab Command

namespace JesseMath.Linter

/-- Compile-time metaprogramming axiom linter (`#assert_zfc_axioms`).
    Scans every theorem and definition elaborated in the current module (`env.constants.map₂`)
    via `Lean.collectAxioms`, verifying that the transitive axiom closure of every
    local declaration is contained strictly within standard ZFC
    (`[propext, Classical.choice, Quot.sound]`).
    Emits zero stdout/info noise when clean, and fails compilation (`throwError`) if any
    declaration depends on an unverified obligation, compiler-trust primitive, or custom axiom. -/
elab "#assert_zfc_axioms" : command => do
  let env ← getEnv
  let allowed : Array Name := #[``propext, ``Classical.choice, ``Quot.sound]
  let mut violations : Array MessageData := #[]
  let mut checkedCount : Nat := 0
  for (declName, cinfo) in env.constants.map₂.toList do
    match cinfo with
    | .axiomInfo _ | .opaqueInfo _ =>
      checkedCount := checkedCount + 1
      let axioms ← liftCoreM (Lean.collectAxioms declName)
      let disallowed := axioms.filter (fun ax => !allowed.contains ax)
      if !disallowed.isEmpty then
        violations := violations.push
          m!"  • {declName} depends on disallowed axioms: {disallowed.toList}"
    | .thmInfo _ | .defnInfo _ =>
      let isUserDecl := !declName.isInternal ||
        (Lean.isPrivateName declName && !(Lean.privateToUserName declName).isInternal)
      if isUserDecl then
        checkedCount := checkedCount + 1
        let axioms ← liftCoreM (Lean.collectAxioms declName)
        let disallowed := axioms.filter (fun ax => !allowed.contains ax)
        if !disallowed.isEmpty then
          violations := violations.push
            m!"  • {declName} depends on disallowed axioms: {disallowed.toList}"
    | _ => pure ()
  if checkedCount == 0 then
    throwError "Axiom audit failed: no local theorem or definition declarations found in module."
  if !violations.isEmpty then
    let body := MessageData.joinSep violations.toList "\n"
    throwError m!"Axiom audit failed ({violations.size} declaration(s) outside ZFC):\n{body}"

end JesseMath.Linter
